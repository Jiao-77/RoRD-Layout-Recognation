import pya
import os

def rasterize_final(gds_path, output_dir, width_px=256):
    # --- 1. 检查与设置 ---
    if not os.path.exists(gds_path):
        print(f"Error: File not found: {gds_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gds_basename = os.path.splitext(os.path.basename(gds_path))[0]
    print(f"Processing: {gds_basename}")

    # --- 2. 加载 Layout ---
    layout = pya.Layout()
    layout.read(gds_path)
    top_cell = layout.top_cell()
    
    if top_cell is None:
        print("Error: No top cell found.")
        return

    # [核心修复] 使用 dbbox() 获取微米(Micron)单位的边框
    # bbox() 返回的是 DBU (Database Units, 整数)，View 可能会把它当做微米导致比例尺错误
    global_dbbox = top_cell.dbbox()
    
    print(f"Global BBox (Microns): {global_dbbox}")
    print(f"Width: {global_dbbox.width()} um, Height: {global_dbbox.height()} um")

    if global_dbbox.width() <= 0:
        print("Error: Layout is empty or zero width.")
        return

    # 计算分辨率
    aspect_ratio = global_dbbox.height() / global_dbbox.width()
    height_px = int(width_px * aspect_ratio)
    height_px = max(1, height_px)

    # --- 3. 初始化视图 ---
    view = pya.LayoutView()
    view.show_layout(layout, False)
    view.max_hier_levels = 1000
    
    # 设置为黑底（用于正式输出）
    view.set_config("background-color", "#000000") 
    view.set_config("grid-visible", "false")

    layer_indices = layout.layer_indices()
    saved_count = 0

    for layer_idx in layer_indices:
        # 检查内容 (注意：bbox_per_layer 也要看情况，这里我们直接渲染不设防)
        # 为了效率，可以先检查该层是否为空
        if top_cell.bbox_per_layer(layer_idx).empty():
            continue

        layer_info = layout.get_info(layer_idx)
        
        # 输出文件名
        filename = f"{gds_basename}_{layer_info.layer}_{layer_info.datatype}.png"
        full_output_path = os.path.join(output_dir, filename)

        # --- 4. 配置图层 ---
        iter = view.begin_layers()
        while not iter.at_end():
            view.delete_layer(iter)

        props = pya.LayerPropertiesNode()
        props.source_layer_index = layer_idx
        
        # 实心填充
        props.dither_pattern = 0 
        
        # 白色填充 + 白色边框
        props.fill_color = 0xFFFFFF 
        props.frame_color = 0xFFFFFF
        
        # 稍微加粗一点边框，保证极细线条也能被渲染
        props.width = 0 
        props.visible = True
        
        view.insert_layer(view.end_layers(), props)

        # [核心修复] 使用微米坐标 Zoom
        view.zoom_box(global_dbbox)

        # 保存
        view.save_image(full_output_path, width_px, height_px)
        print(f"Saved: {filename}")
        saved_count += 1

    print(f"Done. Generated {saved_count} images.")

if __name__ == "__main__":
    # 请替换为你的实际路径
    input_gds = "/home/jiao77/Documents/data/ICCAD2019/layout/patid_MX_Benchmark2_clip_hotspot1_11_orig_0.gds"
    output_folder = "out/final_images"
    resolution_width = 256
    
    rasterize_final(input_gds, output_folder, resolution_width)