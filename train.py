# train.py

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.ic_dataset import ICLayoutTrainingDataset
from losses import compute_detection_loss, compute_description_loss
from models.rord import RoRD
from utils.config_loader import load_config, to_absolute_path
from utils.data_utils import get_transform

# 设置日志记录
def setup_logging(save_dir):
    """设置训练日志记录"""
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

# --- (已修改) 主函数与命令行接口 ---
def main(args):
    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent

    data_dir = args.data_dir or str(to_absolute_path(cfg.paths.layout_dir, config_dir))
    save_dir = args.save_dir or str(to_absolute_path(cfg.paths.save_dir, config_dir))
    epochs = args.epochs if args.epochs is not None else int(cfg.training.num_epochs)
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.training.batch_size)
    lr = args.lr if args.lr is not None else float(cfg.training.learning_rate)
    patch_size = int(cfg.training.patch_size)
    scale_range = tuple(float(x) for x in cfg.training.scale_jitter_range)

    logger = setup_logging(save_dir)

    logger.info("--- 开始训练 RoRD 模型 ---")
    logger.info(f"训练参数: Epochs={epochs}, Batch Size={batch_size}, LR={lr}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"保存目录: {save_dir}")

    transform = get_transform()

    dataset = ICLayoutTrainingDataset(
        data_dir,
        patch_size=patch_size,
        transform=transform,
        scale_range=scale_range,
    )
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = RoRD().cuda()
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 早停机制
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # 训练阶段
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
            
            # 梯度裁剪，防止梯度爆炸
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
        
        # 验证阶段
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
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        logger.info(f"--- Epoch {epoch+1} 完成 ---")
        logger.info(f"训练 - Total: {avg_train_loss:.4f}, Det: {avg_det_loss:.4f}, Desc: {avg_desc_loss:.4f}")
        logger.info(f"验证 - Total: {avg_val_loss:.4f}, Det: {avg_val_det_loss:.4f}, Desc: {avg_val_desc_loss:.4f}")
        logger.info(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, 'rord_model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'config_path': str(Path(args.config).resolve()),
                }
            }, save_path)
            logger.info(f"最佳模型已保存至: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"早停触发！{patience} 个epoch没有改善")
                break
    
    # 保存最终模型
    save_path = os.path.join(save_dir, 'rord_model_final.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_loss': avg_val_loss,
        'config': {
            'learning_rate': lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'config_path': str(Path(args.config).resolve()),
        }
    }, save_path)
    logger.info(f"最终模型已保存至: {save_path}")
    logger.info("训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 RoRD 模型")
    parser.add_argument('--config', type=str, default="configs/base_config.yaml", help="YAML 配置文件路径")
    parser.add_argument('--data_dir', type=str, default=None, help="训练数据目录，若未提供则使用配置文件中的路径")
    parser.add_argument('--save_dir', type=str, default=None, help="模型保存目录，若未提供则使用配置文件中的路径")
    parser.add_argument('--epochs', type=int, default=None, help="训练轮数，若未提供则使用配置文件中的值")
    parser.add_argument('--batch_size', type=int, default=None, help="批次大小，若未提供则使用配置文件中的值")
    parser.add_argument('--lr', type=float, default=None, help="学习率，若未提供则使用配置文件中的值")
    main(parser.parse_args())