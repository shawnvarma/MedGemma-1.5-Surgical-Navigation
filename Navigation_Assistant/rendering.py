"""Image rendering -- overlays, crosshairs, arrows, and 3-D views.

Deduplication:
  * ``_render_overlay()`` replaces three copy-pasted overlay loops.
  * ``draw_arrow()`` merges ``draw_arrow_on_image`` and
    ``draw_target_with_arrow`` into one function.
"""

import json
import math

import numpy as np
from PIL import Image, ImageDraw

try:
    import plotly.graph_objects as go
    from skimage import measure
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly or scikit-image not installed. 3D view disabled.")

from config import TISSUE_LABELS, DISPLAY_SIZE, SCALE
from state import STATE


# ===================================================================
# Shared helpers (dedup)
# ===================================================================

def _render_overlay(img: Image.Image, seg_slice: np.ndarray) -> Image.Image:
    """Composite a segmentation overlay onto a grayscale image.

    ``seg_slice`` must already be oriented to match ``img`` (same height
    and width).  Tissue labels are drawn at 40 / 255 alpha.
    """
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_px = overlay.load()

    for label, info in TISSUE_LABELS.items():
        if label == 0:
            continue
        mask = seg_slice == label
        color = info["color"]
        rows, cols = np.where(mask)
        for r, c in zip(rows, cols):
            overlay_px[c, r] = (color[0], color[1], color[2], 40)

    return Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')


def draw_arrow(image: Image.Image, from_disp: tuple, to_disp: tuple,
                color: tuple = (0, 255, 0)) -> Image.Image:
    """Draw a dashed arrow with crosshair at the target.

    Both *from_disp* and *to_disp* are in **display** pixel coordinates.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    fx, fy = from_disp
    tx, ty = to_disp
    dark = (color[0] // 2, color[1] // 2, color[2] // 2)

    length = math.sqrt((tx - fx)**2 + (ty - fy)**2)
    if length > 10:
        dash_len, gap_len = 10, 6
        dx = (tx - fx) / length
        dy = (ty - fy) / length
        pos = 0
        while pos < length - 15:
            seg_end = min(pos + dash_len, length - 15)
            sx1, sy1 = fx + dx * pos, fy + dy * pos
            sx2, sy2 = fx + dx * seg_end, fy + dy * seg_end
            draw.line([(sx1, sy1), (sx2, sy2)], fill=color, width=2)
            pos += dash_len + gap_len

        # Arrowhead
        arrow_len, arrow_w = 12, 8
        bx = tx - dx * arrow_len
        by = ty - dy * arrow_len
        px, py_ = -dy, dx
        draw.polygon([
            (tx, ty),
            (bx + px * arrow_w / 2, by + py_ * arrow_w / 2),
            (bx - px * arrow_w / 2, by - py_ * arrow_w / 2),
        ], fill=color)

    # Crosshair at target
    size, thick = 12, 3
    draw.line([(tx - size - 1, ty), (tx + size + 1, ty)],
              fill=dark, width=thick + 2)
    draw.line([(tx, ty - size - 1), (tx, ty + size + 1)],
              fill=dark, width=thick + 2)
    draw.line([(tx - size, ty), (tx + size, ty)], fill=color, width=thick)
    draw.line([(tx, ty - size), (tx, ty + size)], fill=color, width=thick)

    radius = 4
    draw.ellipse([tx - radius, ty - radius, tx + radius, ty + radius],
                 fill=color, outline=dark)

    return img_copy


def draw_arrow_to_target(image: Image.Image,
                         from_x: int, from_y: int,
                         to_x: int, to_y: int,
                         scale: float) -> Image.Image:
    """Convenience wrapper: convert native coords to display, then draw."""
    return draw_arrow(
        image,
        (int(from_x * scale), int(from_y * scale)),
        (int(to_x * scale), int(to_y * scale)),
    )


# ===================================================================
# Crosshair
# ===================================================================

def draw_crosshair(image: Image.Image, x: int, y: int,
                   size: int = 10, color: str = 'red',
                   thickness: int = 3) -> Image.Image:
    """Draw a crosshair marker at *(x, y)* on a copy of *image*."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    draw.line([(x - size, y), (x + size, y)], fill=color, width=thickness)
    draw.line([(x, y - size), (x, y + size)], fill=color, width=thickness)
    return img


# ===================================================================
# Slice rendering
# ===================================================================

def get_trajectory_marked_image(z: int, x: int, y: int) -> Image.Image:
    """Slice image with a red crosshair for trajectory analysis input."""
    if STATE.volume is None:
        return Image.new('RGB', (240, 240), color=(50, 50, 50))

    s = STATE.volume[:, :, z].T
    s = s - s.min()
    if s.max() > 0:
        s = (s / s.max() * 255).astype(np.uint8)
    else:
        s = s.astype(np.uint8)
    img = Image.fromarray(s, mode='L').convert('RGB')
    return draw_crosshair(img, x, y, size=10, color='red', thickness=3)


def get_slice_image(z: int, crosshair_pos=None, show_overlay: bool = False,
                    show_distances: bool = True,
                    distances: dict = None) -> Image.Image:
    """Render an axial slice at display resolution with optional overlays."""
    if STATE.volume is None:
        return Image.new('RGB', (DISPLAY_SIZE, DISPLAY_SIZE),
                         color=(50, 50, 50))

    s = STATE.volume[:, :, z].T
    s = s - s.min()
    if s.max() > 0:
        s = (s / s.max() * 255).astype(np.uint8)
    else:
        s = s.astype(np.uint8)
    img = Image.fromarray(s, mode='L').convert('RGB')

    if show_overlay and STATE.segmentation is not None:
        img = _render_overlay(img, STATE.segmentation[:, :, z].T)

    native_size = img.size[0]
    img = img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)
    scale = DISPLAY_SIZE / native_size

    # Legend
    if show_overlay and STATE.segmentation is not None:
        _draw_legend(img)

    # Crosshair and distance lines
    if crosshair_pos is not None:
        cx, cy = crosshair_pos
        cx_d = int(cx * scale)
        cy_d = int(cy * scale)
        draw = ImageDraw.Draw(img)

        if show_distances and distances and distances.get("inside_tissue"):
            _draw_distance_lines(draw, cx_d, cy_d, distances, scale)

        size = 15
        draw.line([(cx_d - size, cy_d), (cx_d + size, cy_d)],
                  fill='red', width=3)
        draw.line([(cx_d, cy_d - size), (cx_d, cy_d + size)],
                  fill='red', width=3)

    return img


def _draw_legend(img: Image.Image):
    """Draw tissue colour legend in the top-left corner."""
    draw = ImageDraw.Draw(img)
    lx, ly = 10, 10
    box, lh = 14, 20
    items = [(1, "NCR/NET (Necrotic)", (255, 0, 0)),
             (2, "Edema", (0, 255, 0)),
             (4, "Enhancing", (255, 255, 0))]
    draw.rectangle([lx - 5, ly - 5,
                    lx + 160, ly + len(items) * lh + 10],
                   fill=(0, 0, 0, 180))
    for i, (_, name, color) in enumerate(items):
        yp = ly + i * lh
        draw.rectangle([lx, yp, lx + box, yp + box],
                       fill=color, outline=(255, 255, 255))
        tx = lx + box + 6
        for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((tx + ox, yp + oy), name, fill=(0, 0, 0))
        draw.text((tx, yp), name, fill=(255, 255, 255))


def _draw_distance_lines(draw, cx, cy, distances, scale):
    """Draw dashed measurement lines from crosshair to tissue edges."""
    dash, gap = 8, 4
    line_color = (0, 0, 0)
    outline = (255, 255, 255)
    r = 4

    def _dashed(x1, y1, x2, y2):
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length == 0:
            return
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        pos = 0
        while pos < length:
            se = min(pos + dash, length)
            sx1, sy1 = x1 + dx * pos, y1 + dy * pos
            sx2, sy2 = x1 + dx * se, y1 + dy * se
            draw.line([(sx1, sy1), (sx2, sy2)], fill=outline, width=4)
            draw.line([(sx1, sy1), (sx2, sy2)], fill=line_color, width=2)
            pos += dash + gap

    def _circle(x, y):
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=outline, outline=line_color, width=2)

    def _label(text, x, y):
        for ox, oy in [(-1, -1), (-1, 1), (1, -1), (1, 1),
                        (-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + ox, y + oy), text, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255))

    if "superior_mm" in distances:
        ey = max(0, cy - int(distances["superior_mm"] * scale))
        _dashed(cx, cy, cx, ey)
        _circle(cx, ey)
        _label(f"{distances['superior_mm']:.0f}mm",
               cx + 8, (cy + ey) // 2 - 6)

    if "inferior_mm" in distances:
        ey = min(DISPLAY_SIZE - 1, cy + int(distances["inferior_mm"] * scale))
        _dashed(cx, cy, cx, ey)
        _circle(cx, ey)
        _label(f"{distances['inferior_mm']:.0f}mm",
               cx + 8, (cy + ey) // 2 - 6)

    if "left_mm" in distances:
        ex = max(0, cx - int(distances["left_mm"] * scale))
        _dashed(cx, cy, ex, cy)
        _circle(ex, cy)
        _label(f"R {distances['left_mm']:.0f}mm",
               (cx + ex) // 2 - 16, cy - 18)

    if "right_mm" in distances:
        ex = min(DISPLAY_SIZE - 1, cx + int(distances["right_mm"] * scale))
        _dashed(cx, cy, ex, cy)
        _circle(ex, cy)
        _label(f"L {distances['right_mm']:.0f}mm",
               (cx + ex) // 2 - 16, cy - 18)


# ===================================================================
# Coronal & Sagittal helpers
# ===================================================================

_MID_COLORS = [(255, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255)]


def _draw_trajectory_on_ortho(draw, trajectory_points, coord_fn, new_h):
    """Draw trajectory markers on a coronal or sagittal view.

    *coord_fn* maps a trajectory point dict to ``(x_disp, z_disp)``.
    """
    if not trajectory_points or len(trajectory_points) == 0:
        return

    if len(trajectory_points) >= 2:
        for i in range(len(trajectory_points) - 1):
            x1, z1 = coord_fn(trajectory_points[i])
            x2, z2 = coord_fn(trajectory_points[i + 1])
            draw.line([(x1, z1), (x2, z2)], fill=(255, 255, 255), width=2)

    for i, pt in enumerate(trajectory_points):
        if i == 0:
            color = (0, 255, 0)
        elif i == len(trajectory_points) - 1:
            color = (255, 0, 0)
        else:
            color = _MID_COLORS[(i - 1) % len(_MID_COLORS)]
        xd, zd = coord_fn(pt)
        r = 6
        draw.ellipse([xd - r - 1, zd - r - 1, xd + r + 1, zd + r + 1],
                     outline=(255, 255, 255), width=2)
        draw.ellipse([xd - r, zd - r, xd + r, zd + r], fill=color)
        draw.text((xd + r + 3, zd - 6), str(i + 1), fill=(255, 255, 255))


def get_coronal_image(y: int, z: int = None, crosshair_x: int = None,
                      show_overlay: bool = False,
                      trajectory_points: list = None) -> Image.Image:
    """Render a coronal (front) view."""
    if STATE.volume is None:
        return Image.new('RGB', (DISPLAY_SIZE, DISPLAY_SIZE // 2),
                         color=(50, 50, 50))

    s = np.flipud(STATE.volume[:, y, :].T)
    s = s - s.min()
    if s.max() > 0:
        s = (s / s.max() * 255).astype(np.uint8)
    else:
        s = s.astype(np.uint8)
    img = Image.fromarray(s, mode='L').convert('RGB')

    if show_overlay and STATE.segmentation is not None:
        seg = np.flipud(STATE.segmentation[:, y, :].T)
        img = _render_overlay(img, seg)

    h, w = s.shape
    new_w = DISPLAY_SIZE
    new_h = int(h * (DISPLAY_SIZE / w))
    img = img.resize((new_w, new_h), Image.Resampling.NEAREST)

    z_depth = STATE.volume.shape[2]
    draw = ImageDraw.Draw(img)
    if z is not None:
        z_d = int(((z_depth - 1) - z) * (new_h / z_depth))
        draw.line([(0, z_d), (new_w, z_d)], fill='red', width=2)
    if crosshair_x is not None:
        draw.line([(int(crosshair_x * SCALE), 0),
                   (int(crosshair_x * SCALE), new_h)],
                  fill='red', width=2)

    def _coord(pt):
        return (int(pt['x'] * SCALE),
                int(((z_depth - 1) - pt['z']) * (new_h / z_depth)))

    _draw_trajectory_on_ortho(draw, trajectory_points, _coord, new_h)
    return img


def get_sagittal_image(x: int, z: int = None, crosshair_y: int = None,
                       show_overlay: bool = False,
                       trajectory_points: list = None) -> Image.Image:
    """Render a sagittal (side) view."""
    if STATE.volume is None:
        return Image.new('RGB', (DISPLAY_SIZE, DISPLAY_SIZE // 2),
                         color=(50, 50, 50))

    s = np.flipud(STATE.volume[x, :, :].T)
    s = s - s.min()
    if s.max() > 0:
        s = (s / s.max() * 255).astype(np.uint8)
    else:
        s = s.astype(np.uint8)
    img = Image.fromarray(s, mode='L').convert('RGB')

    if show_overlay and STATE.segmentation is not None:
        seg = np.flipud(STATE.segmentation[x, :, :].T)
        img = _render_overlay(img, seg)

    h, w = s.shape
    new_w = DISPLAY_SIZE
    new_h = int(h * (DISPLAY_SIZE / w))
    img = img.resize((new_w, new_h), Image.Resampling.NEAREST)

    z_depth = STATE.volume.shape[2]
    draw = ImageDraw.Draw(img)
    if z is not None:
        z_d = int(((z_depth - 1) - z) * (new_h / z_depth))
        draw.line([(0, z_d), (new_w, z_d)], fill='red', width=2)
    if crosshair_y is not None:
        draw.line([(int(crosshair_y * SCALE), 0),
                   (int(crosshair_y * SCALE), new_h)],
                  fill='red', width=2)

    def _coord(pt):
        return (int(pt['y'] * SCALE),
                int(((z_depth - 1) - pt['z']) * (new_h / z_depth)))

    _draw_trajectory_on_ortho(draw, trajectory_points, _coord, new_h)
    return img


# ===================================================================
# Model input image helper
# ===================================================================

def create_crosshair_image_for_model(z: int, x: int,
                                     y: int) -> Image.Image:
    """MRI slice with a red crosshair for model tissue classification."""
    s = STATE.volume[:, :, z].T
    s = np.clip(s, 0, np.percentile(s, 99))
    s = (s / max(s.max(), 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(s).convert("RGB")
    draw = ImageDraw.Draw(img)
    sz, c = 5, (255, 0, 0)
    draw.line([(x - sz, y), (x + sz, y)], fill=c, width=2)
    draw.line([(x, y - sz), (x, y + sz)], fill=c, width=2)
    return img


# ===================================================================
# 3-D Segmentation View (Plotly)
# ===================================================================

def create_3d_segmentation_view(current_z=None, crosshair_pos=None,
                                trajectory_points=None,
                                selected_point_idx=None):
    """Interactive 3-D Plotly view of the tumor segmentation.

    Returns a ``go.Figure`` or ``None`` if Plotly is unavailable.
    """
    if not PLOTLY_AVAILABLE:
        return None

    # Normalise trajectory_points from various Gradio state formats
    if isinstance(trajectory_points, str):
        try:
            trajectory_points = json.loads(trajectory_points)
        except json.JSONDecodeError:
            trajectory_points = []
    if isinstance(trajectory_points, dict) and 'points' in trajectory_points:
        trajectory_points = trajectory_points['points']
    if (trajectory_points and isinstance(trajectory_points, list)
            and trajectory_points and isinstance(trajectory_points[0], str)):
        try:
            trajectory_points = [
                json.loads(p) if isinstance(p, str) else p
                for p in trajectory_points]
        except json.JSONDecodeError:
            trajectory_points = []

    if STATE.segmentation is None:
        fig = go.Figure()
        fig.add_annotation(text="No segmentation data loaded",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="white"))
        fig.update_layout(paper_bgcolor='rgb(30,30,30)',
                          plot_bgcolor='rgb(30,30,30)')
        return fig

    seg = STATE.segmentation
    tissue_colors = {
        1: ('rgb(255, 100, 100)', 'Necrotic Core'),
        2: ('rgb(100, 255, 100)', 'Edema'),
        4: ('rgb(255, 255, 100)', 'Enhancing Tumor'),
    }

    fig = go.Figure()

    # Downsample for performance
    step = 2
    seg_small = seg[::step, ::step, ::step]

    for label, (color, name) in tissue_colors.items():
        mask = (seg_small == label).astype(float)
        if not np.any(mask):
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
            verts = verts * step
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, opacity=0.1, name=name,
                hoverinfo='skip', showlegend=True))
        except Exception:
            pass

    # Brain surface for anatomical context
    if STATE.volume is not None:
        try:
            vol_small = STATE.volume[::step, ::step, ::step]
            threshold = np.percentile(vol_small[vol_small > 0], 5)
            brain = (vol_small > threshold).astype(float)
            verts, faces, _, _ = measure.marching_cubes(brain, level=0.5)
            verts = verts * step
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='rgb(200, 200, 220)', opacity=0.10, name='Brain',
                hoverinfo='skip', showlegend=True))
        except Exception:
            pass

    # Current slice indicator
    if current_z is not None:
        h, w = seg.shape[0], seg.shape[1]
        fig.add_trace(go.Scatter3d(
            x=[w, w, None, 0, w],
            y=[0, h, None, h, h],
            z=[current_z] * 2 + [None] + [current_z] * 2,
            mode='lines',
            line=dict(color='rgba(100, 150, 255, 0.8)', width=3),
            name=f'Slice {current_z}', hoverinfo='name', showlegend=True))

    # Crosshair marker
    if crosshair_pos is not None and current_z is not None:
        fig.add_trace(go.Scatter3d(
            x=[crosshair_pos[0]], y=[crosshair_pos[1]], z=[current_z],
            mode='markers',
            marker=dict(size=8, color='red', symbol='cross'),
            name='Query Position', hoverinfo='name', showlegend=True))

    # Trajectory
    if trajectory_points and len(trajectory_points) > 0:
        tx = [p['x'] for p in trajectory_points]
        ty = [p['y'] for p in trajectory_points]
        tz = [p['z'] for p in trajectory_points]

        if len(trajectory_points) >= 2:
            fig.add_trace(go.Scatter3d(
                x=tx, y=ty, z=tz, mode='lines',
                line=dict(color='white', width=3),
                name='Trajectory Path', hoverinfo='name', showlegend=True))

        mid_c = ['rgb(255,255,0)', 'rgb(255,165,0)',
                 'rgb(255,0,255)', 'rgb(0,255,255)']
        colors, labels = [], []
        for i, pt in enumerate(trajectory_points):
            status = _tissue_status_char(pt['x'], pt['y'], pt['z'])
            n = i + 1
            if i == 0:
                colors.append('lime')
                labels.append(f'1{status} Entry')
            elif i == len(trajectory_points) - 1:
                colors.append('red')
                labels.append(f'{n}{status} Target')
            else:
                colors.append(mid_c[(i - 1) % len(mid_c)])
                labels.append(f'{n}{status}')

        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz, mode='markers+text',
            marker=dict(size=12, color=colors, symbol='diamond',
                        line=dict(color='white', width=1)),
            text=labels, textposition='top center',
            textfont=dict(color='white', size=11),
            hoverinfo='name', name='Trajectory Points', showlegend=True))

    # Annotation for selected point
    scene_annots = []
    if (selected_point_idx is not None and trajectory_points
            and 0 <= selected_point_idx < len(trajectory_points)):
        pt = trajectory_points[selected_point_idx]
        tissue = pt.get('tissue', 'Unknown')
        is_el = pt.get('is_eloquent', False)
        el_info = pt.get('eloquent_info')
        analysis = pt.get('medgemma_analysis', '')

        if not analysis and STATE.last_distilled_per_slice:
            analysis = STATE.last_distilled_per_slice.get(pt.get('z'), "")

        status = ("ok" if tissue in ['NCR_NET', 'ENHANCING']
                  else ("!" if tissue == 'EDEMA' else "X"))
        plabel = ("Entry" if selected_point_idx == 0
                  else ("Target"
                        if selected_point_idx == len(trajectory_points) - 1
                        else f"Point {selected_point_idx + 1}"))
        lines = [f"<b>{plabel} \u2014 Slice {pt.get('z', '?')}: "
                 f"{tissue}</b> {status}"]
        if analysis:
            lines.append("")
            lines.append("<b>[Distillation Fine-Tune]</b>")
            lines.append(f"<i>{_wrap(analysis, 45)}</i>")
        if is_el and el_info:
            lines.append(f"<b>!! ELOQUENT:</b> {el_info}")

        border = ('lime' if status == "ok"
                  else ('yellow' if status == "!" else 'red'))
        scene_annots.append(dict(
            x=pt['x'], y=pt['y'], z=pt['z'],
            text="<br>".join(lines), showarrow=True,
            arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='white',
            ax=240, ay=-200,
            font=dict(size=11, color='white'),
            bgcolor='rgba(0,0,0,0.92)', bordercolor=border,
            borderwidth=2, borderpad=8))

    # Layout
    axis = dict(
        gridcolor='rgba(80,80,80,0.3)', showbackground=False,
        linecolor='rgba(80,80,80,0.5)',
        tickfont=dict(color='rgba(150,150,150,0.7)', size=10))
    tf = dict(color='rgba(150,150,150,0.7)', size=11)
    scene = dict(
        xaxis=dict(title=dict(text='X', font=tf), **axis),
        yaxis=dict(title=dict(text='Y', font=tf), **axis),
        zaxis=dict(title=dict(text='Slice (Z)', font=tf), **axis),
        aspectmode='data',
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0)),
        bgcolor='rgb(30,30,30)')
    if scene_annots:
        scene['annotations'] = scene_annots

    fig.update_layout(
        scene=scene,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor='rgba(50,50,50,0.8)',
                    font=dict(color='white')),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgb(30,30,30)',
        title=dict(text='3D Tumor Segmentation',
                   font=dict(color='white', size=14), x=0.5),
        modebar=dict(bgcolor='rgba(50,50,50,0.8)', color='white',
                     activecolor='rgb(100,150,255)'))

    return fig


# ===================================================================
# 3-D view helpers
# ===================================================================

def _tissue_status_char(x, y, z):
    """Unicode status marker for a voxel: ok / ! / X."""
    if STATE.segmentation is None:
        return "[?]"
    seg_slice = STATE.segmentation[:, :, z].T
    h, w = seg_slice.shape
    cx = max(0, min(int(x), w - 1))
    cy = max(0, min(int(y), h - 1))
    label = int(seg_slice[cy, cx])
    if label in [1, 4]:
        return "\u2713"
    if label == 2:
        return "!"
    return "\u2717"


def _wrap(text, width=45):
    """Wrap text for Plotly annotations (using ``<br>``)."""
    words = text.split()
    lines, cur, cur_len = [], [], 0
    for w in words:
        if cur_len + len(w) + 1 <= width:
            cur.append(w)
            cur_len += len(w) + 1
        else:
            if cur:
                lines.append(' '.join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(' '.join(cur))
    return '<br>'.join(lines)
