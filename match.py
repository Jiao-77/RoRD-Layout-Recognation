# match.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import argparse
import os

import config
from models.rord import RoRD
from utils.data_utils import get_transform

# --- Feature extraction functions (unchanged) ---
def extract_keypoints_and_descriptors(model, image_tensor, kp_thresh):
    with torch.no_grad():
        detection_map, desc = model(image_tensor)
    
    device = detection_map.device
    binary_map = (detection_map > kp_thresh).squeeze(0).squeeze(0)
    coords = torch.nonzero(binary_map).float() # y, x
    
    if len(coords) == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device)

    # Descriptor sampling
    coords_for_grid = coords.flip(1).view(1, -1, 1, 2) # N, 2 -> 1, N, 1, 2 (x,y)
    # Normalize to [-1, 1]
    coords_for_grid = coords_for_grid / torch.tensor([(desc.shape[3]-1)/2, (desc.shape[2]-1)/2], device=device) - 1
    
    descriptors = F.grid_sample(desc, coords_for_grid, align_corners=True).squeeze().T
    descriptors = F.normalize(descriptors, p=2, dim=1)
    
    # Convert keypoint coordinates from feature map scale back to image scale
    # VGG downsampling rate to relu4_3 is 8
    keypoints = coords.flip(1) * 8.0 # x, y

    return keypoints, descriptors

# --- (New) Sliding window feature extraction function ---
def extract_features_sliding_window(model, large_image, transform):
    """
    Extract all keypoints and descriptors from large image using sliding window
    """
    print("Using sliding window to extract features from large layout...")
    device = next(model.parameters()).device
    W, H = large_image.size
    window_size = config.INFERENCE_WINDOW_SIZE
    stride = config.INFERENCE_STRIDE

    all_kps = []
    all_descs = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # Ensure window does not exceed boundaries
            x_end = min(x + window_size, W)
            y_end = min(y + window_size, H)
            
            # Crop window
            patch = large_image.crop((x, y, x_end, y_end))
            
            # Preprocess
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            
            # Extract features
            kps, descs = extract_keypoints_and_descriptors(model, patch_tensor, config.KEYPOINT_THRESHOLD)
            
            if len(kps) > 0:
                # Convert local coordinates to global coordinates
                kps[:, 0] += x
                kps[:, 1] += y
                all_kps.append(kps)
                all_descs.append(descs)
    
    if not all_kps:
        return torch.tensor([], device=device), torch.tensor([], device=device)

    print(f"Large layout feature extraction completed, found {sum(len(k) for k in all_kps)} keypoints in total.")
    return torch.cat(all_kps, dim=0), torch.cat(all_descs, dim=0)


# --- Mutual nearest neighbor matching (unchanged) ---
def mutual_nearest_neighbor(descs1, descs2):
    if len(descs1) == 0 or len(descs2) == 0:
        return torch.empty((0, 2), dtype=torch.int64)
    sim = descs1 @ descs2.T
    nn12 = torch.max(sim, dim=1)
    nn21 = torch.max(sim, dim=0)
    ids1 = torch.arange(0, sim.shape[0], device=sim.device)
    mask = (ids1 == nn21.indices[nn12.indices])
    matches = torch.stack([ids1[mask], nn12.indices[mask]], dim=1)
    return matches

# --- (Modified) Multi-scale, multi-instance matching main function ---
def match_template_multiscale(model, layout_image, template_image, transform):
    """
    Search for template at different scales and detect multiple instances
    """
    # 1. Use sliding window to extract all features from large layout
    layout_kps, layout_descs = extract_features_sliding_window(model, layout_image, transform)
    
    if len(layout_kps) < config.MIN_INLIERS:
        print("Too few keypoints extracted from large layout, cannot perform matching.")
        return []

    found_instances = []
    active_layout_mask = torch.ones(len(layout_kps), dtype=bool, device=layout_kps.device)
    
    # 2. Multi-instance iterative detection
    while True:
        current_active_indices = torch.nonzero(active_layout_mask).squeeze(1)
        
        # Stop if remaining active keypoints are too few
        if len(current_active_indices) < config.MIN_INLIERS:
            break

        current_layout_kps = layout_kps[current_active_indices]
        current_layout_descs = layout_descs[current_active_indices]
        
        best_match_info = {'inliers': 0, 'H': None, 'src_pts': None, 'dst_pts': None, 'mask': None}

        # 3. Image pyramid: iterate through each scale of template
        print("Searching for template at new scale...")
        for scale in config.PYRAMID_SCALES:
            W, H = template_image.size
            new_W, new_H = int(W * scale), int(H * scale)
            
            # Scale template
            scaled_template = template_image.resize((new_W, new_H), Image.LANCZOS)
            template_tensor = transform(scaled_template).unsqueeze(0).to(layout_kps.device)
            
            # Extract features from scaled template
            template_kps, template_descs = extract_keypoints_and_descriptors(model, template_tensor, config.KEYPOINT_THRESHOLD)
            
            if len(template_kps) < 4: continue

            # Match current scale template with active layout features
            matches = mutual_nearest_neighbor(template_descs, current_layout_descs)
            
            if len(matches) < 4: continue

            # RANSAC
            # Note: template keypoint coordinates need to be restored to original size to calculate correct H
            src_pts = template_kps[matches[:, 0]].cpu().numpy() / scale
            dst_pts_indices = current_active_indices[matches[:, 1]]
            dst_pts = layout_kps[dst_pts_indices].cpu().numpy()

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, config.RANSAC_REPROJ_THRESHOLD)

            if H is not None and mask.sum() > best_match_info['inliers']:
                best_match_info = {'inliers': mask.sum(), 'H': H, 'mask': mask, 'scale': scale, 'dst_pts': dst_pts}

        # 4. If best match found across all scales, record and mask
        if best_match_info['inliers'] > config.MIN_INLIERS:
            print(f"Found a matching instance! Inliers: {best_match_info['inliers']}, Template scale used: {best_match_info['scale']:.2f}x")
            
            inlier_mask = best_match_info['mask'].ravel().astype(bool)
            inlier_layout_kps = best_match_info['dst_pts'][inlier_mask]

            x_min, y_min = inlier_layout_kps.min(axis=0)
            x_max, y_max = inlier_layout_kps.max(axis=0)
            
            instance = {'x': int(x_min), 'y': int(y_min), 'width': int(x_max - x_min), 'height': int(y_max - y_min), 'homography': best_match_info['H']}
            found_instances.append(instance)

            # Mask keypoints in matched region to detect next instance
            kp_x, kp_y = layout_kps[:, 0], layout_kps[:, 1]
            region_mask = (kp_x >= x_min) & (kp_x <= x_max) & (kp_y >= y_min) & (kp_y <= y_max)
            active_layout_mask[region_mask] = False
            
            print(f"Remaining active keypoints: {active_layout_mask.sum()}")
        else:
            # If no good match found across all scales, end search
            print("No new matching instances found across all scales, search ended.")
            break
            
    return found_instances


def visualize_matches(layout_path, bboxes, output_path):
    layout_img = cv2.imread(layout_path)
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        cv2.rectangle(layout_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(layout_img, f"Match {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(output_path, layout_img)
    print(f"Visualization result saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-scale template matching using RoRD")
    parser.add_argument('--model_path', type=str, default=config.MODEL_PATH)
    parser.add_argument('--layout', type=str, required=True)
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    transform = get_transform()
    model = RoRD().cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    layout_image = Image.open(args.layout).convert('L')
    template_image = Image.open(args.template).convert('L')
    
    detected_bboxes = match_template_multiscale(model, layout_image, template_image, transform)
    
    print("\nDetected bounding boxes:")
    for bbox in detected_bboxes:
        print(bbox)

    if args.output:
        visualize_matches(args.layout, detected_bboxes, args.output)