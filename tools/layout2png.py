#!/usr/bin/env python3
"""
Batch convert GDS to PNG.

Priority:
1) Use KLayout in headless batch mode (most accurate view fidelity for IC layouts).
2) Fallback to gdstk(read) -> write SVG -> cairosvg to PNG (no KLayout dependency at runtime).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import cairosvg


def klayout_convert(gds_path: Path, png_path: Path, dpi: int, layermap: str | None = None, line_width: int | None = None, bgcolor: str | None = None) -> bool:
    """Render using KLayout by invoking a temporary Python macro with paths embedded."""
    # Prepare optional display config code
    layer_cfg_code = ""
    if layermap:
        # layermap format: "LAYER/DATATYPE:#RRGGBB,..."
        layer_cfg_code += "lprops = pya.LayerPropertiesNode()\n"
        for spec in layermap.split(","):
            spec = spec.strip()
            if not spec:
                continue
            try:
                ld, color = spec.split(":")
                layer_s, datatype_s = ld.split("/")
                color = color.strip()
                layer_cfg_code += (
                    "lp = pya.LayerPropertiesNode()\n"
                    f"lp.layer = int({int(layer_s)})\n"
                    f"lp.datatype = int({int(datatype_s)})\n"
                    f"lp.fill_color = pya.Color.from_string('{color}')\n"
                    f"lp.frame_color = pya.Color.from_string('{color}')\n"
                    "lprops.insert(lp)\n"
                )
            except Exception:
                # Ignore malformed entries
                continue
        layer_cfg_code += "cv.set_layer_properties(lprops)\n"

    line_width_code = ""
    if line_width is not None:
        line_width_code = f"cv.set_config('default-draw-line-width', '{int(line_width)}')\n"

    bg_code = ""
    if bgcolor:
        bg_code = f"cv.set_config('background-color', '{bgcolor}')\n"

    script = f"""
import pya
ly = pya.Layout()
ly.read(r"{gds_path}")
cv = pya.LayoutView()
cv.load_layout(ly, 0)
cv.max_hier_levels = 20
{bg_code}
{line_width_code}
{layer_cfg_code}
cv.zoom_fit()
cv.save_image(r"{png_path}", {dpi}, 0)
"""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
            tf.write(script)
            tf.flush()
            macro_path = Path(tf.name)
        # Run klayout in batch mode
        res = subprocess.run(["klayout", "-zz", "-b", "-r", str(macro_path)], check=False, capture_output=True, text=True)
        ok = res.returncode == 0 and png_path.exists()
        if not ok:
            # Print stderr for visibility when running manually
            if res.stderr:
                sys.stderr.write(res.stderr)
        try:
            macro_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        return ok
    except FileNotFoundError:
        # klayout command not found
        return False
    except Exception:
        return False


def gdstk_fallback(gds_path: Path, png_path: Path, dpi: int) -> bool:
    """Fallback path: use gdstk to read GDS and write SVG, then cairosvg to PNG.
    Note: This may differ visually from KLayout depending on layers/styles.
    """
    try:
        import gdstk  # local import to avoid import cost when not needed
        svg_path = png_path.with_suffix(".svg")
        lib = gdstk.read_gds(str(gds_path))
        tops = lib.top_level()
        if not tops:
            return False
        # Combine tops into a single temporary cell for rendering
        cell = tops[0]
        # gdstk Cell has write_svg in recent versions
        try:
            cell.write_svg(str(svg_path))  # type: ignore[attr-defined]
        except Exception:
            # Older gdstk: write_svg available on Library
            try:
                lib.write_svg(str(svg_path))  # type: ignore[attr-defined]
            except Exception:
                return False
        # Convert SVG to PNG
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=dpi)
        try:
            svg_path.unlink()
        except Exception:
            pass
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert GDS files to PNG")
    parser.add_argument("--in", dest="in_dir", type=str, required=True, help="Input directory containing .gds files")
    parser.add_argument("--out", dest="out_dir", type=str, required=True, help="Output directory to place .png files")
    parser.add_argument("--dpi", type=int, default=600, help="Output resolution in DPI for rasterization")
    parser.add_argument("--layermap", type=str, default=None, help="Layer color map, e.g. '1/0:#00FF00,2/0:#FF0000'")
    parser.add_argument("--line_width", type=int, default=None, help="Default draw line width in pixels for KLayout display")
    parser.add_argument("--bgcolor", type=str, default=None, help="Background color, e.g. '#000000' or 'black'")

    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gds_files = sorted(in_dir.glob("*.gds"))
    if not gds_files:
        print(f"[WARN] No GDS files found in {in_dir}")
        return

    ok_cnt = 0
    for gds in gds_files:
        png_path = out_dir / (gds.stem + ".png")
        ok = klayout_convert(gds, png_path, args.dpi, layermap=args.layermap, line_width=args.line_width, bgcolor=args.bgcolor)
        if not ok:
            ok = gdstk_fallback(gds, png_path, args.dpi)
        if ok:
            ok_cnt += 1
            print(f"[OK] {gds.name} -> {png_path}")
        else:
            print(f"[FAIL] {gds.name}")
    print(f"Done. {ok_cnt}/{len(gds_files)} converted.")


if __name__ == "__main__":
    main()
