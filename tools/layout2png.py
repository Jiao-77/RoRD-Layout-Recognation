import gdstk
import cairosvg
import argparse
import os

def convert_layout_to_png_via_svg(layout_path, png_path, cell_name=None, pixels_per_unit=10):
    """
    通过先生成 SVG 再转换为 PNG 的方式，将 GDSII 或 OASIS 文件光栅化。
    此版本修正了 write_svg 的参数错误，兼容性更强。

    参数:
        layout_path (str): 输入的版图文件路径（.gds 或 .oas）。
        png_path (str): 输出的 PNG 文件路径。
        cell_name (str, optional): 需要转换的单元名称。如果为 None，则使用顶层单元。
        pixels_per_unit (int, optional): 版图数据库单位到像素的转换比例，控制图像分辨率。
    """
    print(f"正在从 '{layout_path}' 读取版图文件...")

    # 1. 加载版图文件
    _, extension = os.path.splitext(layout_path)
    extension = extension.lower()

    if extension == '.gds':
        lib = gdstk.read_gds(layout_path)
    elif extension == '.oas':
        lib = gdstk.read_oas(layout_path)
    else:
        raise ValueError(f"不支持的文件类型: '{extension}'。请输入 .gds 或 .oas 文件。")

    if cell_name:
        cell = lib.cells[cell_name]
    else:
        top_cells = lib.top_level()
        if not top_cells:
            raise ValueError("错误：版图文件中没有找到顶层单元。")
        cell = top_cells[0]
        print(f"未指定单元名称，自动选择顶层单元: '{cell.name}'")

    # 2. 将版图单元写入临时的 SVG 文件 (已移除无效的 padding 参数)
    temp_svg_path = png_path + ".temp.svg"
    print(f"步骤 1/2: 正在将单元 '{cell.name}' 转换为临时 SVG 文件...")
    cell.write_svg(
        temp_svg_path # 隐藏默认字体，避免影响边界
    )

    # 3. 使用 cairosvg 将 SVG 文件转换为 PNG
    print(f"步骤 2/2: 正在将 SVG 转换为 PNG...")
    # 获取单元的精确边界框
    bb = cell.bb()
    if bb is None:
        raise ValueError(f"单元 '{cell.name}' 为空或无法获取其边界框。")
        
    # 根据边界框和分辨率计算输出图像的宽度
    width, height = bb[1] - bb[0]
    output_width = width * pixels_per_unit
    
    cairosvg.svg2png(url=temp_svg_path, write_to=png_path, output_width=output_width)

    # 4. 清理临时的 SVG 文件
    os.remove(temp_svg_path)
    
    print(f"成功！图像已保存至: '{png_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 GDSII (.gds) 或 OASIS (.oas) 版图文件转换为 PNG 图像 (通过SVG)。",
        epilog="示例: python rasterize.py -i my_chip.oas -o my_chip.png -ppu 20"
    )
    parser.add_argument('-i', '--input', type=str, required=True, help="输入的版图文件路径 (.gds 或 .oas)。")
    parser.add_argument('-o', '--output', type=str, help="输出的 PNG 文件路径。如果未提供，将使用输入文件名并替换扩展名为 .png。")
    parser.add_argument('-c', '--cell', type=str, default=None, help="要转换的特定单元的名称。默认为顶层单元。")
    parser.add_argument('-ppu', '--pixels_per_unit', type=int, default=10, help="每微米（um）的像素数，用于控制输出图像的分辨率。")
    args = parser.parse_args()

    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}.png"
        print(f"未指定输出路径，将自动保存为: '{args.output}'")

    try:
        convert_layout_to_png_via_svg(
            layout_path=args.input,
            png_path=args.output,
            cell_name=args.cell,
            pixels_per_unit=args.pixels_per_unit
        )
    except Exception as e:
        print(f"\n处理失败: {e}")