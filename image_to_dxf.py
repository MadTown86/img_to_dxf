"""
image_to_dxf.py  —  PNG/JPG → Closed Contour DXF for SolidWorks multi-color 3D printing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE (in VSCode terminal or any command prompt):

  Basic:
      python image_to_dxf.py image.png

  With options:
      python image_to_dxf.py image.png --colors 10 --min-area 250 --scale 0.1

  Custom output name/folder:
      python image_to_dxf.py image.png --output my_drawing.dxf

  No preview image:
      python image_to_dxf.py image.png --no-preview

  Full example:
      python image_to_dxf.py BASS_PNG.png --colors 10 --blur 2 --min-area 250 --simplify 0.0015 --arc-tol 3.0 --scale 0.1

PARAMETER GUIDE:
  --colors     Number of color regions to separate  (default: 10,  range: 2-20)
  --blur       Gaussian blur radius to reduce noise  (default: 2,   range: 0-6)
  --min-area   Minimum region size in pixels to keep (default: 250, range: 20-5000)
  --simplify   Contour simplification factor         (default: 0.0015, lower = more detail)
  --arc-tol    Arc fitting tolerance in pixels       (default: 3.0, higher = more arcs)
  --scale      Output scale: mm per pixel            (default: 0.1 → 1px = 0.1mm)
  --max-size   Internal processing resolution cap    (default: 800px on longest side)

REQUIREMENTS:
  pip install opencv-python numpy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import numpy as np
import math
import sys
import time
import argparse
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
#  TERMINAL COLOURS  (works on Windows 10+, macOS, Linux)
# ══════════════════════════════════════════════════════════════════════

class C:
    """ANSI colour helpers."""
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    RED    = "\033[91m"
    GREY   = "\033[90m"
    WHITE  = "\033[97m"

    @staticmethod
    def ok(msg):    return f"{C.GREEN}✔  {msg}{C.RESET}"
    @staticmethod
    def step(msg):  return f"{C.CYAN}▶  {msg}{C.RESET}"
    @staticmethod
    def warn(msg):  return f"{C.YELLOW}⚠  {msg}{C.RESET}"
    @staticmethod
    def err(msg):   return f"{C.RED}✘  {msg}{C.RESET}"
    @staticmethod
    def info(msg):  return f"{C.GREY}   {msg}{C.RESET}"
    @staticmethod
    def head(msg):  return f"\n{C.BOLD}{C.WHITE}{msg}{C.RESET}"


def progress_bar(current, total, width=30, label=""):
    """Print an in-place progress bar."""
    frac  = current / max(total, 1)
    filled = int(width * frac)
    bar   = "█" * filled + "░" * (width - filled)
    pct   = int(frac * 100)
    print(f"\r  {C.CYAN}[{bar}]{C.RESET} {pct:3d}%  {C.GREY}{label}{C.RESET}   ", end="", flush=True)


def print_banner():
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════╗
║          IMAGE → CLOSED CONTOUR DXF GENERATOR       ║
║        For SolidWorks Multi-Color 3D Printing        ║
╚══════════════════════════════════════════════════════╝{C.RESET}""")


# ══════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════

def circ3(p1, p2, p3):
    """Circumcircle of three points. Returns (cx, cy, r) or None."""
    ax, ay = p2[0] - p1[0], p2[1] - p1[1]
    bx, by = p3[0] - p1[0], p3[1] - p1[1]
    D = 2 * (ax * by - ay * bx)
    if abs(D) < 1e-8:
        return None
    a2 = ax*ax + ay*ay
    b2 = bx*bx + by*by
    ux = (by*a2 - ay*b2) / D
    uy = (ax*b2 - bx*a2) / D
    return (p1[0]+ux, p1[1]+uy, math.sqrt(ux*ux + uy*uy))


def get_bulge(p1, p2, pm):
    """DXF LWPOLYLINE bulge value for the arc p1→p2 passing through pm."""
    c = circ3(p1, pm, p2)
    if not c or c[2] < 0.5:
        return 0.0
    cx, cy, r = c
    chord = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    if chord < 0.5 or chord > 2*r + 0.01:
        return 0.0
    sin_h = min(1.0, chord / (2*r))
    try:
        bv = math.tan(math.asin(sin_h) / 2)
    except Exception:
        return 0.0
    cross = (p2[0]-p1[0])*(pm[1]-p1[1]) - (p2[1]-p1[1])*(pm[0]-p1[0])
    return bv * (1 if cross >= 0 else -1) if math.isfinite(bv) else 0.0


def add_bulges(simp, orig, arc_tol):
    """
    For each edge of the simplified polygon, try to replace it with an arc.
    simp : [[x,y], ...]  simplified vertices
    orig : [[x,y], ...]  full boundary pixels
    Returns: [{'x':..., 'y':..., 'bulge':...}, ...]
    """
    n, m = len(simp), len(orig)
    if m < 6:
        return [{'x': p[0], 'y': p[1], 'bulge': 0.0} for p in simp]

    # Map each simplified vertex back to the nearest original index
    orig_idx = []
    sf = 0
    for sp in simp:
        best_i, best_d = sf, 1e18
        for di in range(m):
            k = (sf + di) % m
            d = (orig[k][0]-sp[0])**2 + (orig[k][1]-sp[1])**2
            if d < best_d:
                best_d = d
                best_i = k
            if d < 1:
                break  # exact match — stop early
        orig_idx.append(best_i)
        sf = best_i

    result = []
    for i in range(n):
        j = (i + 1) % n
        a, b = orig_idx[i], orig_idx[j]

        # Collect original pixels between vertex i and vertex j
        between = []
        k = (a + 1) % m
        safety = 0
        while k != b and safety < m:
            between.append(orig[k])
            k = (k + 1) % m
            safety += 1

        bulge = 0.0
        if len(between) >= 4:
            mid = between[len(between) // 2]
            c = circ3(simp[i], simp[j], mid)
            if c and c[2] > 0.5:
                cx, cy, r = c
                errs = [abs(math.hypot(p[0]-cx, p[1]-cy) - r) for p in between]
                if max(errs) <= arc_tol:
                    bulge = get_bulge(simp[i], simp[j], mid)

        result.append({'x': simp[i][0], 'y': simp[i][1], 'bulge': bulge})

    return result


# ══════════════════════════════════════════════════════════════════════
#  DXF WRITER
# ══════════════════════════════════════════════════════════════════════

def build_dxf(layers, W, H, scale):
    """
    Produce an AC1015 DXF string with one closed LWPOLYLINE per contour.
    Y-axis is flipped so the drawing is right-side-up in CAD (0,0 = bottom-left).
    """
    L = []
    hdl = [200]

    def H_():
        hdl[0] += 1
        return format(hdl[0]-1, 'X').zfill(3)

    # AutoCAD Color Index values — each layer gets a distinct colour in CAD
    ACI = [1,2,3,4,5,6,7,30,40,50,60,70,80,90,100,110,120,130,140,150]

    # ── HEADER ──────────────────────────────────────────────────────
    L += [
        '  0','SECTION','  2','HEADER',
        '  9','$ACADVER','  1','AC1015',
        '  9','$INSBASE',' 10','0.0',' 20','0.0',' 30','0.0',
        '  9','$EXTMIN',' 10','0.0',' 20','0.0',' 30','0.0',
        '  9','$EXTMAX',
            ' 10', f'{W*scale:.4f}',
            ' 20', f'{H*scale:.4f}',
            ' 30', '0.0',
        '  9','$INSUNITS',' 70','4',   # 4 = millimetres
        '  0','ENDSEC',
    ]

    # ── TABLES ──────────────────────────────────────────────────────
    L += [
        '  0','SECTION','  2','TABLES',
        '  0','TABLE','  2','LTYPE','  5',H_(),'100','AcDbSymbolTable',' 70','1',
        '  0','LTYPE','  5',H_(),'100','AcDbSymbolTableRecord',
            '100','AcDbLinetypeTableRecord',
            '  2','Continuous',' 70','0','  3','Solid',' 72','65',' 73','0',' 40','0.0',
        '  0','ENDTAB',
        '  0','TABLE','  2','LAYER','  5',H_(),'100','AcDbSymbolTable',
            ' 70', str(len(layers)),
    ]
    for i, lay in enumerate(layers):
        L += [
            '  0','LAYER','  5',H_(),
            '100','AcDbSymbolTableRecord','100','AcDbLayerTableRecord',
            '  2', lay['name'],
            ' 70','0',
            ' 62', str(ACI[i % len(ACI)]),
            '  6','Continuous',
        ]
    L += ['  0','ENDTAB','  0','ENDSEC']

    # ── ENTITIES — one closed LWPOLYLINE per contour ─────────────────
    L += ['  0','SECTION','  2','ENTITIES']
    for lay in layers:
        for cont in lay['contours']:
            if len(cont) < 3:
                continue
            L += [
                '  0','LWPOLYLINE','  5',H_(),
                '100','AcDbEntity','  8', lay['name'],
                '100','AcDbPolyline',
                ' 90', str(len(cont)),
                ' 70','1',            # closed flag
            ]
            for v in cont:
                # Flip Y so drawing origin is bottom-left (standard CAD)
                L += [' 10', f"{v['x']*scale:.5f}",
                      ' 20', f"{(H - v['y'])*scale:.5f}"]
                if abs(v.get('bulge', 0)) > 1e-6:
                    L += [' 42', f"{v['bulge']:.7f}"]

    L += ['  0','ENDSEC','  0','EOF']
    return '\n'.join(L)


# ══════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════

def process(input_path, output_dxf, output_preview,
            blur=2, n_colors=10, min_area=250,
            dp_eps=0.0015, arc_tol=3.0, scale=0.1, max_size=800):

    t_start = time.time()

    # ── 1. LOAD IMAGE ───────────────────────────────────────────────
    print(C.step("Loading image..."))
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        print(C.err(f"Could not open image: {input_path}"))
        print(C.info("Supported formats: PNG, JPG, BMP, TIFF, WebP"))
        sys.exit(1)

    H0, W0 = img_bgr.shape[:2]
    s = min(max_size / W0, max_size / H0, 1.0)
    W = int(W0 * s)
    H = int(H0 * s)

    print(C.info(f"Original size : {W0} × {H0} px"))
    if s < 1.0:
        print(C.info(f"Processing at : {W} × {H} px  (scaled to {max_size}px cap)"))
    else:
        print(C.info(f"Processing at : {W} × {H} px  (full resolution)"))

    img = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA) if s < 1.0 else img_bgr.copy()

    # ── 2. BLUR ─────────────────────────────────────────────────────
    if blur > 0:
        print(C.step(f"Applying blur (radius={blur})..."))
        img = cv2.GaussianBlur(img, (blur*2+1, blur*2+1), 0)
    else:
        print(C.info("Blur skipped (radius=0)"))

    # ── 3. COLOUR QUANTISATION (K-Means) ────────────────────────────
    print(C.step(f"Quantising to {n_colors} colour regions (K-Means)..."))
    data = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels_flat, centers = cv2.kmeans(
        data, n_colors, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    labels_map = labels_flat.reshape(H, W)
    print(C.ok("Colour quantisation complete"))

    # ── 4. CONTOUR EXTRACTION ────────────────────────────────────────
    print(C.step("Extracting and simplifying contours..."))
    all_layers       = []
    total_contours   = 0
    total_arcs       = 0
    total_verts      = 0

    for ci in range(n_colors):
        progress_bar(ci, n_colors, label=f"colour {ci+1}/{n_colors}")

        mask = np.uint8(labels_map == ci) * 255
        b_, g_, r_ = int(centers[ci, 0]), int(centers[ci, 1]), int(centers[ci, 2])  # BGR
        r_rgb, g_rgb, b_rgb = r_, g_, b_  # flip to RGB
        hex_c      = f'{r_rgb:02X}{g_rgb:02X}{b_rgb:02X}'
        layer_name = f'CLR_{hex_c}'

        n_comp, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

        layer_conts = []
        for comp_i in range(1, n_comp):
            area = stats[comp_i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            comp_mask = np.uint8(cc_labels == comp_i) * 255
            # CHAIN_APPROX_NONE keeps every boundary pixel — required for arc fitting
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for cnt in cnts:
                cnt_sq = cnt.squeeze()
                if cnt_sq.ndim < 2 or len(cnt_sq) < 6:
                    continue
                orig = cnt_sq.tolist()

                # Douglas-Peucker simplification
                eps_px = max(1.0, dp_eps * cv2.arcLength(cnt, True))
                approx = cv2.approxPolyDP(cnt, eps_px, True)
                approx_sq = approx.squeeze()
                if approx_sq.ndim < 2 or len(approx_sq) < 3:
                    continue
                simp = approx_sq.tolist()

                # Replace straight edges with arc bulges where possible
                with_bulges = add_bulges(simp, orig, arc_tol)
                layer_conts.append(with_bulges)

                total_contours += 1
                arc_ct = sum(1 for v in with_bulges if abs(v.get('bulge', 0)) > 1e-6)
                total_arcs  += arc_ct
                total_verts += len(with_bulges)

        if layer_conts:
            all_layers.append({
                'name':     layer_name,
                'rgb':      (r_rgb, g_rgb, b_rgb),
                'hex':      hex_c,
                'contours': layer_conts,
            })

    progress_bar(n_colors, n_colors, label="done")
    print()  # newline after progress bar
    print(C.ok("Contour extraction complete"))

    # ── 5. WRITE DXF ─────────────────────────────────────────────────
    print(C.step("Writing DXF file..."))
    dxf_text = build_dxf(all_layers, W, H, scale)
    with open(output_dxf, 'w', encoding='utf-8') as f:
        f.write(dxf_text)
    print(C.ok(f"DXF saved  →  {output_dxf}"))

    # ── 6. WRITE PREVIEW PNG ─────────────────────────────────────────
    if output_preview:
        print(C.step("Rendering preview image..."))
        preview = (img.astype(float) * 0.25 + 245 * 0.75).astype(np.uint8).copy()
        for lay in all_layers:
            r, g, b = lay['rgb']
            col_bgr = (max(0, b-60), max(0, g-60), max(0, r-60))
            for cont in lay['contours']:
                pts = np.array([[int(v['x']), int(v['y'])] for v in cont], dtype=np.int32)
                cv2.polylines(preview, [pts], isClosed=True, color=col_bgr, thickness=1)
        cv2.imwrite(str(output_preview), preview)
        print(C.ok(f"Preview saved  →  {output_preview}"))

    # ── 7. SUMMARY ───────────────────────────────────────────────────
    elapsed = time.time() - t_start
    arc_pct = int(100 * total_arcs / total_verts) if total_verts else 0
    dim_mm  = f"{W*scale:.1f} × {H*scale:.1f} mm"

    print(f"""
{C.BOLD}{C.GREEN}━━━━━━━━━━━━━━━━━━━━━  RESULTS  ━━━━━━━━━━━━━━━━━━━━━{C.RESET}
  {C.WHITE}DXF layers    :{C.RESET}  {len(all_layers)}  (one per unique colour region)
  {C.WHITE}Closed contours:{C.RESET} {total_contours}  (all LWPOLYLINE closed=1)
  {C.WHITE}Total vertices:{C.RESET}  {total_verts:,}
  {C.WHITE}Arc segments  :{C.RESET}  {total_arcs:,}  ({arc_pct}% of edges are arcs)
  {C.WHITE}Drawing size  :{C.RESET}  {dim_mm}  (at {scale} mm/px)
  {C.WHITE}File size     :{C.RESET}  {len(dxf_text)/1024:.1f} KB
  {C.WHITE}Time elapsed  :{C.RESET}  {elapsed:.1f}s
{C.BOLD}{C.GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}

  {C.CYAN}SolidWorks tip:{C.RESET}
  File ▸ Open ▸ select DXF ▸ open as 2D sketch.
  Select all contours ▸ Boss-Extrude ▸ use "Multi-body" option.
  Export as STL ▸ import into Bambu Studio ▸ split objects.
""")

    return all_layers


# ══════════════════════════════════════════════════════════════════════
#  CLI ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog        = "image_to_dxf",
        description = (
            "Convert a PNG/JPG image into a closed-contour DXF file.\n"
            "Each colour region becomes a separate closed LWPOLYLINE layer,\n"
            "ready for SolidWorks extrusion and Bambu Labs multi-colour 3D printing."
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = """
EXAMPLES:
  python image_to_dxf.py fish.png
  python image_to_dxf.py fish.png --colors 12 --scale 0.2
  python image_to_dxf.py fish.png --output ./output/fish.dxf --no-preview
  python image_to_dxf.py logo.png --colors 6 --blur 1 --min-area 150 --simplify 0.001

PARAMETER TIPS:
  Complex images (lots of detail)  →  --colors 12-16  --simplify 0.001  --min-area 150
  Simple logos / cartoon images    →  --colors 6-8    --simplify 0.002  --min-area 300
  Noisy photos / gradients         →  --blur 3        --colors 8        --min-area 500
  Maximum arc substitution         →  --arc-tol 5.0
  Maximum vertex detail            →  --arc-tol 1.0   --simplify 0.0005
        """,
    )

    parser.add_argument(
        "input",
        type    = str,
        help    = "Path to input image (PNG, JPG, BMP, TIFF, WebP)",
    )
    parser.add_argument(
        "--output", "-o",
        type    = str,
        default = None,
        help    = "Output DXF path (default: same folder as input, same name + .dxf)",
    )
    parser.add_argument(
        "--no-preview",
        action  = "store_true",
        help    = "Skip saving the PNG preview image",
    )
    parser.add_argument(
        "--colors", "-c",
        type    = int,
        default = 10,
        metavar = "N",
        help    = "Number of colour regions to extract  [default: 10, range: 2-20]",
    )
    parser.add_argument(
        "--blur", "-b",
        type    = int,
        default = 2,
        metavar = "R",
        help    = "Gaussian blur radius to reduce noise  [default: 2, range: 0-6]",
    )
    parser.add_argument(
        "--min-area", "-a",
        type    = int,
        default = 250,
        metavar = "PX",
        help    = "Minimum region area in pixels to include  [default: 250]",
    )
    parser.add_argument(
        "--simplify", "-s",
        type    = float,
        default = 0.0015,
        metavar = "F",
        help    = "Douglas-Peucker simplification factor (fraction of perimeter)  [default: 0.0015]",
    )
    parser.add_argument(
        "--arc-tol", "-t",
        type    = float,
        default = 3.0,
        metavar = "PX",
        help    = "Arc fitting tolerance in pixels — higher = more arc segments  [default: 3.0]",
    )
    parser.add_argument(
        "--scale",
        type    = float,
        default = 0.1,
        metavar = "MM",
        help    = "Output scale: millimetres per pixel  [default: 0.1]",
    )
    parser.add_argument(
        "--max-size",
        type    = int,
        default = 800,
        metavar = "PX",
        help    = "Cap longest image side before processing  [default: 800]",
    )

    return parser


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def main():
    # Enable ANSI colours on Windows
    if sys.platform == "win32":
        import os
        os.system("color")

    print_banner()

    parser = build_parser()
    args   = parser.parse_args()

    # ── Validate input ───────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        print(C.err(f"Input file not found: {input_path}"))
        sys.exit(1)
    if input_path.suffix.lower() not in {'.png','.jpg','.jpeg','.bmp','.tiff','.tif','.webp'}:
        print(C.warn(f"Unexpected file extension '{input_path.suffix}' — will attempt anyway."))

    # ── Validate numeric ranges ──────────────────────────────────────
    if not (2 <= args.colors <= 20):
        print(C.err("--colors must be between 2 and 20")); sys.exit(1)
    if not (0 <= args.blur <= 6):
        print(C.err("--blur must be between 0 and 6")); sys.exit(1)
    if args.min_area < 1:
        print(C.err("--min-area must be >= 1")); sys.exit(1)
    if args.scale <= 0:
        print(C.err("--scale must be > 0")); sys.exit(1)

    # ── Resolve output paths ─────────────────────────────────────────
    if args.output:
        output_dxf = Path(args.output)
    else:
        output_dxf = input_path.with_suffix('.dxf')

    output_dxf.parent.mkdir(parents=True, exist_ok=True)

    if args.no_preview:
        output_preview = None
    else:
        output_preview = output_dxf.with_suffix('.preview.png')

    # ── Print config summary ─────────────────────────────────────────
    print(C.head("Configuration"))
    print(C.info(f"Input image   : {input_path}"))
    print(C.info(f"Output DXF    : {output_dxf}"))
    if output_preview:
        print(C.info(f"Preview image : {output_preview}"))
    print(C.info(f"Colours       : {args.colors}"))
    print(C.info(f"Blur radius   : {args.blur}"))
    print(C.info(f"Min area      : {args.min_area} px²"))
    print(C.info(f"Simplification: {args.simplify}"))
    print(C.info(f"Arc tolerance : {args.arc_tol} px"))
    print(C.info(f"Scale         : {args.scale} mm/px"))
    print(C.info(f"Max resolution: {args.max_size} px"))

    print(C.head("Processing"))

    # ── Run ──────────────────────────────────────────────────────────
    process(
        input_path     = input_path,
        output_dxf     = output_dxf,
        output_preview = output_preview,
        blur           = args.blur,
        n_colors       = args.colors,
        min_area       = args.min_area,
        dp_eps         = args.simplify,
        arc_tol        = args.arc_tol,
        scale          = args.scale,
        max_size       = args.max_size,
    )


if __name__ == "__main__":
    main()
