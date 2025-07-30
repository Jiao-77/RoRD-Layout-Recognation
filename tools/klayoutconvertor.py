# tools/klayoutconvertor.py
#!/usr/bin/env python3
"""
KLayout GDS to PNG Converter

This script uses KLayout's Python API to convert GDS files to PNG images.
It accepts command-line arguments for input parameters.

Requirements:
    pip install klayout

Usage:
    python klayoutconvertor.py input.gds output.png [options]
"""

import klayout.db as pya
import klayout.lay as lay
from PIL import Image
import os
import argparse
import sys

Image.MAX_IMAGE_PIXELS = None


def export_gds_as_image(
    gds_path: str,
    output_path: str,
    layers: list = [1, 2],
    center_um: tuple = (0, 0),
    view_size_um: float = 100.0,
    resolution: int = 2048,
    binarize: bool = True
) -> None:
    """
    Export GDS file as PNG image using KLayout.
    
    Args:
        gds_path: Input GDS file path
        output_path: Output PNG file path
        layers: List of layer numbers to include
        center_um: Center coordinates in micrometers (x, y)
        view_size_um: View size in micrometers
        resolution: Output image resolution
        binarize: Whether to convert to black and white
    """
    if not os.path.exists(gds_path):
        raise FileNotFoundError(f"Input file not found: {gds_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    layout = pya.Layout()
    layout.read(gds_path)
    top = layout.top_cell()

    # Create layout view
    view = lay.LayoutView()
    view.set_config("background-color", "#ffffff")
    view.set_config("grid-visible", "false")

    # Load layout into view correctly
    view.load_layout(gds_path)
    
    # Add all layers
    view.add_missing_layers()
    
    # Configure view to show entire layout with reasonable resolution
    if view_size_um > 0:
        # Use specified view size
        box = pya.DBox(
            center_um[0] - view_size_um / 2,
            center_um[1] - view_size_um / 2,
            center_um[0] + view_size_um / 2,
            center_um[1] + view_size_um / 2
        )
    else:
        # Use full layout bounds with size limit
        bbox = top.bbox()
        if bbox:
            # Convert to micrometers (KLayout uses database units)
            dbu = layout.dbu
            box = pya.DBox(
                bbox.left * dbu,
                bbox.bottom * dbu,
                bbox.right * dbu,
                bbox.top * dbu
            )
            
        else:
            # Fallback to 100x100 um if empty layout
            box = pya.DBox(-50, -50, 50, 50)

    view.max_hier()
    view.zoom_box(box)
    
    # Save to temporary file first, then load with PIL
    import tempfile
    temp_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    
    try:
        view.save_image(temp_path, resolution, resolution)
        img = Image.open(temp_path)
        
        if binarize:
            # Convert to grayscale and binarize
            img = img.convert("L")
            img = img.point(lambda x: 255 if x > 128 else 0, '1')
        else:
            # Convert to grayscale
            img = img.convert("L")
        
        img.save(output_path)
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Convert GDS to PNG using KLayout')
    parser.add_argument('input', help='Input GDS file')
    parser.add_argument('output', help='Output PNG file')
    parser.add_argument('--layers', nargs='+', type=int, default=[1, 2],
                       help='Layers to include (default: 1 2)')
    parser.add_argument('--center-x', type=float, default=0,
                       help='Center X coordinate in micrometers (default: 0)')
    parser.add_argument('--center-y', type=float, default=0,
                       help='Center Y coordinate in micrometers (default: 0)')
    parser.add_argument('--size', type=float, default=0,
                       help='View size in micrometers (default: 0 = full layout)')
    parser.add_argument('--resolution', type=int, default=2048,
                       help='Output image resolution (default: 2048)')
    parser.add_argument('--no-binarize', action='store_true',
                       help='Disable binarization (keep grayscale)')
    
    args = parser.parse_args()
    
    try:
        export_gds_as_image(
            gds_path=args.input,
            output_path=args.output,
            layers=args.layers,
            center_um=(args.center_x, args.center_y),
            view_size_um=args.size,
            resolution=args.resolution,
            binarize=not args.no_binarize
        )
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()