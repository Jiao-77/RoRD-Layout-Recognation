# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import os
import argparse
import logging
from datetime import datetime

# Import project modules
import config
from models.rord import RoRD
from utils.data_utils import get_transform

# Setup logging
def setup_logging(save_dir):
    """Setup training logging"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# --- (Modified) Training-specific dataset class ---
class ICLayoutTrainingDataset(Dataset):
    def __init__(self, image_dir, patch_size=256, transform=None, scale_range=(1.0, 1.0)):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.patch_size = patch_size
        self.transform = transform
        self.scale_range = scale_range # New scale range parameter

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('L')
        W, H = image.size

        # --- New: Scale jittering data augmentation ---
        # 1. Randomly select a scaling factor
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        # 2. Calculate crop size from original image based on scaling factor
        crop_size = int(self.patch_size / scale)

        # 确保裁剪尺寸不超过图像边界
        if crop_size > min(W, H):
            crop_size = min(W, H)
        
        # 3. Random cropping
        x = np.random.randint(0, W - crop_size + 1)
        y = np.random.randint(0, H - crop_size + 1)
        patch = image.crop((x, y, x + crop_size, y + crop_size))

        # 4. Resize cropped patch back to standard patch_size
        patch = patch.resize((self.patch_size, self.patch_size), Image.Resampling.LANCZOS)
        # --- Scale jittering end ---

        # --- New: Additional data augmentation ---
        # Brightness adjustment
        if np.random.random() < 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            patch = patch.point(lambda x: int(x * brightness_factor))
        
        # Contrast adjustment
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)
            patch = patch.point(lambda x: int(((x - 128) * contrast_factor) + 128))
        
        # Add noise
        if np.random.random() < 0.3:
            patch_np = np.array(patch, dtype=np.float32)
            noise = np.random.normal(0, 5, patch_np.shape)
            patch_np = np.clip(patch_np + noise, 0, 255)
            patch = Image.fromarray(patch_np.astype(np.uint8))
        # --- Additional data augmentation end ---

        patch_np = np.array(patch)
        
        # Implement 8-direction discrete geometric transformations (this logic remains unchanged)
        theta_deg = np.random.choice([0, 90, 180, 270])
        is_mirrored = np.random.choice([True, False])
        cx, cy = self.patch_size / 2.0, self.patch_size / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), theta_deg, 1)

        if is_mirrored:
            T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
            Flip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
            T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
            M_mirror_3x3 = T2 @ Flip @ T1
            M_3x3 = np.vstack([M, [0, 0, 1]])
            H = (M_3x3 @ M_mirror_3x3).astype(np.float32)
        else:
            H = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        
        transformed_patch_np = cv2.warpPerspective(patch_np, H, (self.patch_size, self.patch_size))
        transformed_patch = Image.fromarray(transformed_patch_np)

        if self.transform:
            patch = self.transform(patch)
            transformed_patch = self.transform(transformed_patch)

        H_tensor = torch.from_numpy(H[:2, :]).float()
        return patch, transformed_patch, H_tensor

# --- (Modified) Feature map transformation and loss functions (improved version) ---
def warp_feature_map(feature_map, H_inv):
    B, C, H, W = feature_map.size()
    grid = F.affine_grid(H_inv, feature_map.size(), align_corners=False).to(feature_map.device)
    return F.grid_sample(feature_map, grid, align_corners=False)

def compute_detection_loss(det_original, det_rotated, H):
    """Improved detection loss: use BCE loss instead of MSE"""
    with torch.no_grad():
        H_inv = torch.inverse(torch.cat([H, torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).repeat(H.shape[0], 1, 1)], dim=1))[:, :2, :]
    warped_det_rotated = warp_feature_map(det_rotated, H_inv)
    
    # Use BCE loss, more suitable for binary classification problems
    bce_loss = F.binary_cross_entropy(det_original, warped_det_rotated)
    
    # Add smooth L1 loss as auxiliary
    smooth_l1_loss = F.smooth_l1_loss(det_original, warped_det_rotated)
    
    return bce_loss + 0.1 * smooth_l1_loss

def compute_description_loss(desc_original, desc_rotated, H, margin=1.0):
    """IC layout-specific geometric-aware descriptor loss: encodes Manhattan geometric features"""
    B, C, H_feat, W_feat = desc_original.size()
    
    # Manhattan geometric-aware sampling: focus on edge and corner regions
    num_samples = 200
    
    # Generate Manhattan-aligned sampling grid (horizontal and vertical priority)
    h_coords = torch.linspace(-1, 1, int(np.sqrt(num_samples)), device=desc_original.device)
    w_coords = torch.linspace(-1, 1, int(np.sqrt(num_samples)), device=desc_original.device)
    
    # Increase sampling density in Manhattan directions
    manhattan_h = torch.cat([h_coords, torch.zeros_like(h_coords)])
    manhattan_w = torch.cat([torch.zeros_like(w_coords), w_coords])
    manhattan_coords = torch.stack([manhattan_h, manhattan_w], dim=1).unsqueeze(0).repeat(B, 1, 1)
    
    # Sample anchor points
    anchor = F.grid_sample(desc_original, manhattan_coords.unsqueeze(1), align_corners=False).squeeze(2).transpose(1, 2)
    
    # Calculate corresponding positive samples
    coords_hom = torch.cat([manhattan_coords, torch.ones(B, manhattan_coords.size(1), 1, device=manhattan_coords.device)], dim=2)
    M_inv = torch.inverse(torch.cat([H, torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).repeat(H.shape[0], 1, 1)], dim=1))
    coords_transformed = (coords_hom @ M_inv.transpose(1, 2))[:, :, :2]
    positive = F.grid_sample(desc_rotated, coords_transformed.unsqueeze(1), align_corners=False).squeeze(2).transpose(1, 2)
    
    # IC layout-specific negative sample strategy: consider repetitive structures
    with torch.no_grad():
        # 1. Geometric-aware negative samples: different regions after Manhattan transformation
        neg_coords = []
        for b in range(B):
            # Generate coordinates after Manhattan transformation (90-degree rotation, etc.)
            angles = [0, 90, 180, 270]
            for angle in angles:
                if angle != 0:
                    theta = torch.tensor([angle * np.pi / 180])
                    rot_matrix = torch.tensor([
                        [torch.cos(theta), -torch.sin(theta), 0],
                        [torch.sin(theta), torch.cos(theta), 0]
                    ])
                    rotated_coords = manhattan_coords[b] @ rot_matrix[:2, :2].T
                    neg_coords.append(rotated_coords)
        
        neg_coords = torch.stack(neg_coords[:B*num_samples//2]).reshape(B, -1, 2)
        
        # 2. Feature space hard negative samples
        negative_candidates = F.grid_sample(desc_rotated, neg_coords, align_corners=False).squeeze(2).transpose(1, 2)
        
        # 3. Manhattan distance constrained hard sample selection
        anchor_expanded = anchor.unsqueeze(2).expand(-1, -1, negative_candidates.size(1), -1)
        negative_expanded = negative_candidates.unsqueeze(1).expand(-1, anchor.size(1), -1, -1)
        
        # Use Manhattan distance instead of Euclidean distance
        manhattan_dist = torch.sum(torch.abs(anchor_expanded - negative_expanded), dim=3)
        hard_indices = torch.topk(manhattan_dist, k=anchor.size(1)//2, largest=False)[1]
        negative = torch.gather(negative_candidates, 1, hard_indices)
    
    # IC layout-specific geometric consistency loss
    # 1. Manhattan direction consistency loss
    manhattan_loss = 0
    for i in range(anchor.size(1)):
        # Calculate geometric consistency in horizontal and vertical directions
        anchor_norm = F.normalize(anchor[:, i], p=2, dim=1)
        positive_norm = F.normalize(positive[:, i], p=2, dim=1)
        
        # Encourage descriptor invariance to Manhattan transformations
        cos_sim = torch.sum(anchor_norm * positive_norm, dim=1)
        manhattan_loss += torch.mean(1 - cos_sim)
    
    # 2. Sparsity regularization (IC layout features are sparse)
    sparsity_loss = torch.mean(torch.abs(anchor)) + torch.mean(torch.abs(positive))
    
    # 3. Binary feature distance (handles binary input)
    binary_loss = torch.mean(torch.abs(torch.sign(anchor) - torch.sign(positive)))
    
    # Combined loss
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=1, reduction='mean')  # Use L1 distance
    geometric_triplet = triplet_loss(anchor, positive, negative)
    
    return geometric_triplet + 0.1 * manhattan_loss + 0.01 * sparsity_loss + 0.05 * binary_loss

# --- (Modified) Main function and command-line interface ---
def main(args):
    # Setup logging
    logger = setup_logging(args.save_dir)
    
    logger.info("--- Starting RoRD model training ---")
    logger.info(f"Training parameters: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Save directory: {args.save_dir}")
    
    transform = get_transform()
    
    # Pass scale jittering range during dataset initialization
    dataset = ICLayoutTrainingDataset(
        args.data_dir, 
        patch_size=config.PATCH_SIZE, 
        transform=transform, 
        scale_range=config.SCALE_JITTER_RANGE
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Split training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = RoRD().cuda()
    logger.info(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping mechanism
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_det_loss = 0
        total_desc_loss = 0
        
        for i, (original, rotated, H) in enumerate(train_dataloader):
            original, rotated, H = original.cuda(), rotated.cuda(), H.cuda()
            
            det_original, desc_original = model(original)
            det_rotated, desc_rotated = model(rotated)
            
            det_loss = compute_detection_loss(det_original, det_rotated, H)
            desc_loss = compute_description_loss(desc_original, desc_rotated, H)
            loss = det_loss + desc_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            total_det_loss += det_loss.item()
            total_desc_loss += desc_loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {i}, Total Loss: {loss.item():.4f}, "
                          f"Det Loss: {det_loss.item():.4f}, Desc Loss: {desc_loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_det_loss = total_det_loss / len(train_dataloader)
        avg_desc_loss = total_desc_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_det_loss = 0
        total_val_desc_loss = 0
        
        with torch.no_grad():
            for original, rotated, H in val_dataloader:
                original, rotated, H = original.cuda(), rotated.cuda(), H.cuda()
                
                det_original, desc_original = model(original)
                det_rotated, desc_rotated = model(rotated)
                
                val_det_loss = compute_detection_loss(det_original, det_rotated, H)
                val_desc_loss = compute_description_loss(desc_original, desc_rotated, H)
                val_loss = val_det_loss + val_desc_loss
                
                total_val_loss += val_loss.item()
                total_val_det_loss += val_det_loss.item()
                total_val_desc_loss += val_desc_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_det_loss = total_val_det_loss / len(val_dataloader)
        avg_val_desc_loss = total_val_desc_loss / len(val_dataloader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        logger.info(f"--- Epoch {epoch+1} completed ---")
        logger.info(f"Training - Total: {avg_train_loss:.4f}, Det: {avg_det_loss:.4f}, Desc: {avg_desc_loss:.4f}")
        logger.info(f"Validation - Total: {avg_val_loss:.4f}, Det: {avg_val_det_loss:.4f}, Desc: {avg_val_desc_loss:.4f}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, 'rord_model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'learning_rate': args.lr,
                    'batch_size': args.batch_size,
                    'epochs': args.epochs
                }
            }, save_path)
            logger.info(f"Best model saved to: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered! No improvement for {patience} epochs")
                break
    
    # Save final model
    save_path = os.path.join(args.save_dir, 'rord_model_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_loss': avg_val_loss,
        'config': {
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }
    }, save_path)
    logger.info(f"Final model saved to: {save_path}")
    logger.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoRD model")
    parser.add_argument('--data_dir', type=str, default=config.LAYOUT_DIR)
    parser.add_argument('--save_dir', type=str, default=config.SAVE_DIR)
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    main(parser.parse_args())