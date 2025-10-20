# train.py

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

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

    logging_cfg = cfg.get("logging", None)
    use_tensorboard = False
    log_dir = None
    experiment_name = None

    if logging_cfg is not None:
        use_tensorboard = bool(logging_cfg.get("use_tensorboard", False))
        log_dir = logging_cfg.get("log_dir", "runs")
        experiment_name = logging_cfg.get("experiment_name", "default")

    if args.disable_tensorboard:
        use_tensorboard = False
    if args.log_dir is not None:
        log_dir = args.log_dir
    if args.experiment_name is not None:
        experiment_name = args.experiment_name

    writer = None
    if use_tensorboard and log_dir:
        log_root = Path(log_dir).expanduser()
        experiment_folder = experiment_name or "default"
        tb_path = log_root / "train" / experiment_folder
        tb_path.parent.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tb_path.as_posix())

    logger = setup_logging(save_dir)

    logger.info("--- 开始训练 RoRD 模型 ---")
    logger.info(f"训练参数: Epochs={epochs}, Batch Size={batch_size}, LR={lr}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"保存目录: {save_dir}")
    if writer:
        logger.info(f"TensorBoard 日志目录: {tb_path}")

    transform = get_transform()

    # 读取增强与合成配置
    augment_cfg = cfg.get("augment", {})
    elastic_cfg = augment_cfg.get("elastic", {}) if augment_cfg else {}
    use_albu = bool(elastic_cfg.get("enabled", False))
    albu_params = {
        "prob": elastic_cfg.get("prob", 0.3),
        "alpha": elastic_cfg.get("alpha", 40),
        "sigma": elastic_cfg.get("sigma", 6),
        "alpha_affine": elastic_cfg.get("alpha_affine", 6),
        "brightness_contrast": bool(augment_cfg.get("photometric", {}).get("brightness_contrast", True)) if augment_cfg else True,
        "gauss_noise": bool(augment_cfg.get("photometric", {}).get("gauss_noise", True)) if augment_cfg else True,
    }

    # 构建真实数据集
    real_dataset = ICLayoutTrainingDataset(
        data_dir,
        patch_size=patch_size,
        transform=transform,
        scale_range=scale_range,
        use_albu=use_albu,
        albu_params=albu_params,
    )

    # 读取合成数据配置（程序化 + 扩散）
    syn_cfg = cfg.get("synthetic", {})
    syn_enabled = bool(syn_cfg.get("enabled", False))
    syn_ratio = float(syn_cfg.get("ratio", 0.0))
    syn_dir = syn_cfg.get("png_dir", None)

    syn_dataset = None
    if syn_enabled and syn_dir:
        syn_dir_path = Path(to_absolute_path(syn_dir, config_dir))
        if syn_dir_path.exists():
            syn_dataset = ICLayoutTrainingDataset(
                syn_dir_path.as_posix(),
                patch_size=patch_size,
                transform=transform,
                scale_range=scale_range,
                use_albu=use_albu,
                albu_params=albu_params,
            )
            if len(syn_dataset) == 0:
                syn_dataset = None
        else:
            logger.warning(f"合成数据目录不存在，忽略: {syn_dir_path}")
            syn_enabled = False

    # 扩散生成数据配置
    diff_cfg = syn_cfg.get("diffusion", {}) if syn_cfg else {}
    diff_enabled = bool(diff_cfg.get("enabled", False))
    diff_ratio = float(diff_cfg.get("ratio", 0.0))
    diff_dir = diff_cfg.get("png_dir", None)
    diff_dataset = None
    if diff_enabled and diff_dir:
        diff_dir_path = Path(to_absolute_path(diff_dir, config_dir))
        if diff_dir_path.exists():
            diff_dataset = ICLayoutTrainingDataset(
                diff_dir_path.as_posix(),
                patch_size=patch_size,
                transform=transform,
                scale_range=scale_range,
                use_albu=use_albu,
                albu_params=albu_params,
            )
            if len(diff_dataset) == 0:
                diff_dataset = None
        else:
            logger.warning(f"扩散数据目录不存在，忽略: {diff_dir_path}")
            diff_enabled = False

    logger.info(
        "真实数据集大小: %d%s%s" % (
            len(real_dataset),
            f", 合成(程序)数据集: {len(syn_dataset)}" if syn_dataset else "",
            f", 合成(扩散)数据集: {len(diff_dataset)}" if diff_dataset else "",
        )
    )

    # 验证集仅使用真实数据，避免评价受合成样本干扰
    train_size = int(0.8 * len(real_dataset))
    val_size = max(len(real_dataset) - train_size, 1)
    real_train_dataset, val_dataset = torch.utils.data.random_split(real_dataset, [train_size, val_size])

    # 训练集：可与合成数据集合并（程序合成 + 扩散）
    datasets = [real_train_dataset]
    weights = []
    names = []
    # 收集各源与期望比例
    n_real = len(real_train_dataset)
    n_real = max(n_real, 1)
    names.append("real")
    # 程序合成
    if syn_dataset is not None and syn_enabled and syn_ratio > 0.0:
        datasets.append(syn_dataset)
        names.append("synthetic")
    # 扩散合成
    if diff_dataset is not None and diff_enabled and diff_ratio > 0.0:
        datasets.append(diff_dataset)
        names.append("diffusion")

    if len(datasets) > 1:
        mixed_train_dataset = ConcatDataset(datasets)
        # 计算各源样本数
        counts = [len(real_train_dataset)]
        if syn_dataset is not None and syn_enabled and syn_ratio > 0.0:
            counts.append(len(syn_dataset))
        if diff_dataset is not None and diff_enabled and diff_ratio > 0.0:
            counts.append(len(diff_dataset))
        # 期望比例：real = 1 - (syn_ratio + diff_ratio)
        target_real = max(0.0, 1.0 - (syn_ratio + diff_ratio))
        target_ratios = [target_real]
        if syn_dataset is not None and syn_enabled and syn_ratio > 0.0:
            target_ratios.append(syn_ratio)
        if diff_dataset is not None and diff_enabled and diff_ratio > 0.0:
            target_ratios.append(diff_ratio)
        # 构建每个样本的权重
        per_source_weights = []
        for count, ratio in zip(counts, target_ratios):
            count = max(count, 1)
            per_source_weights.append(ratio / count)
        # 展开到每个样本
        weights = []
        idx = 0
        for count, w in zip(counts, per_source_weights):
            weights += [w] * count
            idx += count
        sampler = WeightedRandomSampler(weights, num_samples=len(mixed_train_dataset), replacement=True)
        train_dataloader = DataLoader(mixed_train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
        logger.info(
            f"启用混采: real={target_real:.2f}, syn={syn_ratio:.2f}, diff={diff_ratio:.2f}; 总样本={len(mixed_train_dataset)}"
        )
        if writer:
            writer.add_text(
                "dataset/mix",
                f"enabled=true, ratios: real={target_real:.2f}, syn={syn_ratio:.2f}, diff={diff_ratio:.2f}; "
                f"counts: real_train={len(real_train_dataset)}, syn={len(syn_dataset) if syn_dataset else 0}, diff={len(diff_dataset) if diff_dataset else 0}"
            )
    else:
        train_dataloader = DataLoader(real_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        if writer:
            writer.add_text("dataset/mix", f"enabled=false, real_train={len(real_train_dataset)}")

    logger.info(f"训练集大小: {len(train_dataloader.dataset)}, 验证集大小: {len(val_dataset)}")
    if writer:
        writer.add_text("dataset/info", f"train={len(train_dataloader.dataset)}, val={len(val_dataset)}")

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

            if writer:
                num_batches = len(train_dataloader) if len(train_dataloader) > 0 else 1
                global_step = epoch * num_batches + i
                writer.add_scalar("train/loss_total", loss.item(), global_step)
                writer.add_scalar("train/loss_det", det_loss.item(), global_step)
                writer.add_scalar("train/loss_desc", desc_loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {i}, Total Loss: {loss.item():.4f}, "
                          f"Det Loss: {det_loss.item():.4f}, Desc Loss: {desc_loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_det_loss = total_det_loss / len(train_dataloader)
        avg_desc_loss = total_desc_loss / len(train_dataloader)
        if writer:
            writer.add_scalar("epoch/train_loss_total", avg_train_loss, epoch)
            writer.add_scalar("epoch/train_loss_det", avg_det_loss, epoch)
            writer.add_scalar("epoch/train_loss_desc", avg_desc_loss, epoch)
        
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
        if writer:
            writer.add_scalar("epoch/val_loss_total", avg_val_loss, epoch)
            writer.add_scalar("epoch/val_loss_det", avg_val_det_loss, epoch)
            writer.add_scalar("epoch/val_loss_desc", avg_val_desc_loss, epoch)
            writer.add_scalar("epoch/lr", optimizer.param_groups[0]['lr'], epoch)
        
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
            if writer:
                writer.add_scalar("checkpoint/best_val_loss", best_val_loss, epoch)
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

    if writer:
        writer.add_scalar("final/val_loss", avg_val_loss, epochs - 1)
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 RoRD 模型")
    parser.add_argument('--config', type=str, default="configs/base_config.yaml", help="YAML 配置文件路径")
    parser.add_argument('--data_dir', type=str, default=None, help="训练数据目录，若未提供则使用配置文件中的路径")
    parser.add_argument('--save_dir', type=str, default=None, help="模型保存目录，若未提供则使用配置文件中的路径")
    parser.add_argument('--epochs', type=int, default=None, help="训练轮数，若未提供则使用配置文件中的值")
    parser.add_argument('--batch_size', type=int, default=None, help="批次大小，若未提供则使用配置文件中的值")
    parser.add_argument('--lr', type=float, default=None, help="学习率，若未提供则使用配置文件中的值")
    parser.add_argument('--log_dir', type=str, default=None, help="TensorBoard 日志根目录，覆盖配置文件中的设置")
    parser.add_argument('--experiment_name', type=str, default=None, help="TensorBoard 实验名称，覆盖配置文件中的设置")
    parser.add_argument('--disable_tensorboard', action='store_true', help="禁用 TensorBoard 日志记录")
    main(parser.parse_args())