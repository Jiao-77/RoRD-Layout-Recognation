import pya
import os
import glob

def batch_rasterize_layer_10_0(input_dir, output_dir, width_px=256):
    # --- 1. 环境准备 ---
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 获取所有 gds 文件 (不区分大小写)
    gds_files = glob.glob(os.path.join(input_dir, "*.gds")) + \
                glob.glob(os.path.join(input_dir, "*.GDS"))
    
    # 去重并排序
    gds_files = sorted(list(set(gds_files)))
    
    total_files = len(gds_files)
    print(f"Found {total_files} GDS files in {input_dir}")
    print("-" * 50)

    # 定义目标层
    TARGET_LAYER = 10
    TARGET_DATATYPE = 0

    # --- 2. 批量处理循环 ---
    for i, gds_path in enumerate(gds_files):
        try:
            gds_filename = os.path.basename(gds_path)
            gds_basename = os.path.splitext(gds_filename)[0]
            
            # 输出文件路径: out_dir/filename.png
            output_path = os.path.join(output_dir, f"{gds_basename}.png")
            
            print(f"[{i+1}/{total_files}] Processing: {gds_filename} ...", end="", flush=True)

            # --- 加载 Layout ---
            layout = pya.Layout()
            layout.read(gds_path)
            top_cell = layout.top_cell()
            
            if top_cell is None:
                print(" -> Error: No Top Cell")
                continue

            # --- 获取微米单位的 BBox (关键修复) ---
            global_dbbox = top_cell.dbbox()
            
            # 如果 BBox 无效，跳过
            if global_dbbox.width() <= 0 or global_dbbox.height() <= 0:
                print(" -> Error: Empty Layout")
                continue

            # --- 计算分辨率 ---
            aspect_ratio = global_dbbox.height() / global_dbbox.width()
            height_px = int(width_px * aspect_ratio)
            height_px = max(1, height_px)

            # --- 初始化视图 ---
            view = pya.LayoutView()
            view.show_layout(layout, False)
            view.max_hier_levels = 1000  # 保证显示所有层级
            
            # 配置背景 (黑底)
            view.set_config("background-color", "#000000") 
            view.set_config("grid-visible", "false")

            # --- 配置 Layer 10/0 ---
            
            # 1. 清除默认图层
            iter = view.begin_layers()
            while not iter.at_end():
                view.delete_layer(iter)

            # 2. 查找目标层索引
            # find_layer 返回索引，如果没找到通常需要在后续判断
            # 注意：即使文件里没有这一层，我们通常也需要生成一张全黑图片以保持数据集完整性
            layer_idx = layout.find_layer(TARGET_LAYER, TARGET_DATATYPE)
            
            # 检查该层是否存在于 layout 中
            if layer_idx is not None:
                # 检查该层在 Top Cell 下是否有内容 (可选，为了效率)
                # 如果你需要即便没内容也输出黑图，可以保留逻辑继续
                
                props = pya.LayerPropertiesNode()
                props.source_layer_index = layer_idx
                
                # --- 沿用你确认可用的参数 ---
                props.dither_pattern = 0   # 你的配置: 0
                props.width = 0            # 你的配置: 0
                props.fill_color = 0xFFFFFF 
                props.frame_color = 0xFFFFFF
                props.visible = True
                
                view.insert_layer(view.end_layers(), props)
            else:
                # 如果没找到层，保持 view 里没有层，结果将是纯黑背景
                # 这在机器学习数据集中通常是期望的行为（Label为空）
                pass

            # --- 锁定视角 (使用 Micron 坐标) ---
            view.zoom_box(global_dbbox)

            # --- 保存图片 ---
            view.save_image(output_path, width_px, height_px)
            print(" Done.")

        except Exception as e:
            print(f" -> Exception: {e}")

    print("-" * 50)
    print("Batch processing finished.")

# --- 主程序入口 ---
if __name__ == "__main__":
    # 配置输入输出文件夹
    input_folder = "/home/jiao77/Documents/data/ICCAD2019/layout" # 你的 GDS 文件夹
    output_folder = "/home/jiao77/Documents/data/ICCAD2019/img"                            # 输出图片文件夹
    resolution_width = 256                                        # 图片宽度
    
    batch_rasterize_layer_10_0(input_folder, output_folder, resolution_width)