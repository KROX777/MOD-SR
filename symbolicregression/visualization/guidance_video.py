import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


class GuidanceSubtreeVideoRecorder:
    """Capture guidance-subtree token-logit evolution and export MP4/GIF."""

    def __init__(
        self,
        output_dir: str,
        fps: int = 2,
        topk: int = 3,
        tree_width_scale: float = 1.8,
        eval_points: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(1, int(fps))
        self.topk = max(1, int(topk))
        self.tree_width_scale = max(1.0, float(tree_width_scale))
        self.eval_points = max(1, int(eval_points))
        self._frame_idx = 0
        self._frame_paths: List[Path] = []
        self._session_prefix: Optional[Path] = None

    def begin(self, objective: str, t_idx: int):
        run_name = f"guidance_{objective}_t{int(t_idx):04d}"
        self._session_prefix = self.output_dir / run_name
        frame_dir = self._session_prefix / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        self._frame_idx = 0
        self._frame_paths = []

    def add_frame(
        self,
        *,
        fex_env,
        logits: torch.Tensor,
        active_seq_len: int,
        active_positions: Optional[Sequence[int]],
        subtree_root: Optional[int],
        t_idx: int,
        inner_step: int,
        phase: str,
        frame_meta: Optional[Dict[str, Any]] = None,
    ):
        if self._session_prefix is None:
            return
        frame_dir = self._session_prefix / "frames"
        frame_path = frame_dir / f"frame_{self._frame_idx:04d}.png"
        render_relaxed_subtree(
            fex_env=fex_env,
            logits=logits,
            active_seq_len=active_seq_len,
            active_positions=active_positions,
            subtree_root=subtree_root,
            output_path=frame_path,
            topk=self.topk,
            tree_width_scale=self.tree_width_scale,
            eval_points=self.eval_points,
            title=f"t={t_idx} | inner={inner_step} | {phase}",
            frame_meta=frame_meta,
        )
        self._frame_paths.append(frame_path)
        self._frame_idx += 1

    def finalize(self) -> Dict[str, Optional[str]]:
        if self._session_prefix is None or len(self._frame_paths) == 0:
            return {"gif": None}
        gif_path = self._session_prefix / "subtree_logits.gif"

        gif_ok = self._build_gif(self._frame_paths, gif_path)

        return {
            "gif": str(gif_path) if gif_ok else None,
        }

    def _build_gif(self, frame_paths: List[Path], out_path: Path) -> bool:
        if len(frame_paths) == 0:
            return False
        try:
            images = [Image.open(p).convert("RGB") for p in frame_paths]
            duration_ms = int(1000 / self.fps)
            images[0].save(
                out_path,
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=0,
            )
            for img in images:
                img.close()
            return out_path.exists()
        except Exception:
            return False


def render_fex_tree(fex_encoder, tree_tokens, expr_str: str, output_path: str):
    """Standalone tree renderer (migrated from fixed_tree_encoder.py)."""
    from graphviz import Digraph

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dot = Digraph(comment=f"FEX Tree: {expr_str}")
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='filled', fontname='Arial')
    dot.attr('edge', color='#555555')

    for i, node in enumerate(fex_encoder.tree.nodes):
        token = tree_tokens[i]
        if isinstance(token, tuple) and len(token) == 2:
            pos1, pos2 = token
            if pos1 == '<PAD>' and pos2 == '<PAD>':
                label = 'PAD'
                color = 'yellow'
            else:
                label = f'{pos1}\\n{pos2}'
                color = 'lightblue'
        elif isinstance(token, str):
            if token.startswith('<ID_Binary>'):
                label = 'ID_Bin'
                color = 'lightgray'
            elif token.startswith('<ID_Unary>'):
                label = 'ID_Una'
                color = 'lightgray'
            else:
                label = token
                color = 'lightcoral'
        else:
            label = str(token)
            color = 'white'
        layer_info = f'\\n(L{node["layer"]}, #{i})'
        dot.node(str(i), label + layer_info, fillcolor=color)

    for i, node in enumerate(fex_encoder.tree.nodes):
        if node['type'] == 'leaf':
            continue
        if node['type'] == 'unary':
            child_layer = node['layer'] + 1
            child_index_in_layer = node['index_in_layer']
            for j, n in enumerate(fex_encoder.tree.nodes):
                if n['layer'] == child_layer and n['index_in_layer'] == child_index_in_layer:
                    dot.edge(str(i), str(j))
                    break
        else:
            child_layer = node['layer'] + 1
            left_index = 2 * node['index_in_layer']
            right_index = 2 * node['index_in_layer'] + 1
            for j, n in enumerate(fex_encoder.tree.nodes):
                if n['layer'] == child_layer and n['index_in_layer'] == left_index:
                    dot.edge(str(i), str(j), label='L')
                    break
            for j, n in enumerate(fex_encoder.tree.nodes):
                if n['layer'] == child_layer and n['index_in_layer'] == right_index:
                    dot.edge(str(i), str(j), label='R')
                    break

    dot.render(str(output_path.with_suffix('')), format=output_path.suffix.lstrip('.') or 'png', cleanup=True)


def render_relaxed_subtree(
    *,
    fex_env,
    logits: torch.Tensor,
    active_seq_len: int,
    active_positions: Optional[Sequence[int]],
    subtree_root: Optional[int],
    output_path: Path,
    topk: int = 3,
    tree_width_scale: float = 1.8,
    eval_points: int = 5,
    title: Optional[str] = None,
    frame_meta: Optional[Dict[str, Any]] = None,
):
    """Render subtree logits snapshot for one inner-loop step."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if logits.dim() != 2:
        raise ValueError("Expected logits shape (seq_len, vocab).")

    encoder = fex_env.fex_encoder
    id2word = getattr(fex_env, "equation_id2word", {})
    seq_cache = encoder._get_seq_pos_cache() if hasattr(encoder, "_get_seq_pos_cache") else None

    def seq_pos(node_idx: int) -> int:
        if seq_cache is not None:
            if isinstance(seq_cache, dict):
                return seq_cache.get(node_idx, -1) + 1
            if 0 <= node_idx < len(seq_cache):
                return seq_cache[node_idx] + 1
            return -1
        return encoder._get_sequence_position(node_idx) + 1

    block_set = set(active_positions or [])
    node_ids: Set[int] = set()
    if len(block_set) > 0:
        for node in encoder.tree.nodes:
            node_idx = node['inorder_idx']
            p = seq_pos(node_idx)
            if p in block_set or (node['type'] == 'leaf' and (p + 1) in block_set):
                node_ids.add(node_idx)
    else:
        node_ids = {n['inorder_idx'] for n in encoder.tree.nodes if seq_pos(n['inorder_idx']) < active_seq_len}

    if len(node_ids) == 0:
        node_ids = {n['inorder_idx'] for n in encoder.tree.nodes[:1]}

    if subtree_root is None:
        if len(node_ids) > 0:
            subtree_root = min(node_ids)
        else:
            subtree_root = encoder.tree.get_root_inorder_idx()

    positions = _layout_positions(encoder, width_scale=tree_width_scale)

    fig = plt.figure(figsize=(24, 13))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.9], hspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[1, 0])
    ax.set_axis_off()
    ax_info.set_axis_off()
    if title:
        ax.set_title(title, fontsize=14)

    # edges
    for node in encoder.tree.nodes:
        parent = node['inorder_idx']
        if parent not in node_ids:
            continue
        children = _children(encoder, node)
        for child in children:
            if child in node_ids:
                x1, y1 = positions[parent]
                x2, y2 = positions[child]
                ax.plot([x1, x2], [y1, y2], color="#777", linewidth=1.2, zorder=1)

    # nodes
    for node in encoder.tree.nodes:
        idx = node['inorder_idx']
        if idx not in node_ids:
            continue
        x, y = positions[idx]
        p = seq_pos(idx)
        color = "#fce8b2" if node['type'] == 'binary' else ("#f8c6d8" if node['type'] == 'unary' else "#cde8ff")
        if idx == subtree_root:
            color = "#8fd3ff"

        ax.scatter([x], [y], s=1400, c=color, edgecolors="#333", linewidths=1.0, zorder=3)

        is_frozen = (len(block_set) > 0) and (p not in block_set)
        lines = [f"{node['type']}#{idx} pos={p}{' [frozen]' if is_frozen else ''}"]
        if 0 <= p < logits.size(0):
            lines.extend(
                _topk_text(
                    logits[p],
                    id2word,
                    topk=topk,
                    position=p,
                    frame_meta=frame_meta,
                )
            )
        if node['type'] == 'leaf' and 0 <= p + 1 < logits.size(0):
            lines.append("--")
            lines.extend(
                _topk_text(
                    logits[p + 1],
                    id2word,
                    topk=topk,
                    position=p + 1,
                    frame_meta=frame_meta,
                )
            )

        ax.text(
            x,
            y,
            "\n".join(lines),
            ha="center",
            va="center",
            fontsize=7.5,
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.55, linewidth=0),
        )

    xs = [positions[i][0] for i in node_ids]
    ys = [positions[i][1] for i in node_ids]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max(0.2, (x_max - x_min) * 0.12)
    y_pad = max(0.2, (y_max - y_min) * 0.15)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    _draw_metadata_panel(
        ax_info=ax_info,
        frame_meta=frame_meta,
        fex_env=fex_env,
        logits=logits,
        active_seq_len=active_seq_len,
        block_set=block_set,
        eval_points=eval_points,
    )

    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _topk_text(
    logits_row: torch.Tensor,
    id2word: Dict[int, str],
    topk: int,
    position: Optional[int] = None,
    frame_meta: Optional[Dict[str, Any]] = None,
) -> List[str]:
    # Handle both torch.Tensor and np.ndarray for robustness
    if hasattr(logits_row, "numel"):
        num_elements = logits_row.numel()
    else:
        num_elements = logits_row.size
    
    topk = max(1, min(topk, num_elements))
    
    if torch.is_tensor(logits_row):
        vals, inds = torch.topk(logits_row, k=topk)
    else:
        # Fallback for numpy if it ever receives one
        logits_row_torch = torch.from_numpy(logits_row)
        vals, inds = torch.topk(logits_row_torch, k=topk)

    probs_topk = torch.softmax(vals, dim=0)
    grad_pos = None
    if frame_meta is not None and position is not None:
        pos_grad_norm = frame_meta.get("pos_grad_norm")
        try:
            if isinstance(pos_grad_norm, torch.Tensor) and 0 <= position < pos_grad_norm.numel():
                grad_pos = float(pos_grad_norm[position].item())
            elif isinstance(pos_grad_norm, np.ndarray) and 0 <= position < pos_grad_norm.size:
                grad_pos = float(pos_grad_norm[position])
            elif isinstance(pos_grad_norm, (list, tuple)) and 0 <= position < len(pos_grad_norm):
                grad_pos = float(pos_grad_norm[position])
        except Exception:
            grad_pos = None

    if frame_meta is not None and position is not None:
        topk_inds_fm = frame_meta.get("topk_inds")
        topk_vals_fm = frame_meta.get("topk_vals")
        topk_probs_fm = frame_meta.get("topk_probs")
        if (
            isinstance(topk_inds_fm, torch.Tensor)
            and isinstance(topk_vals_fm, torch.Tensor)
            and isinstance(topk_probs_fm, torch.Tensor)
            and 0 <= position < topk_inds_fm.size(0)
        ):
            inds = topk_inds_fm[position][:topk]
            vals = topk_vals_fm[position][:topk]
            probs_topk = topk_probs_fm[position][:topk]

    lines = []
    for j in range(topk):
        idx = int(inds[j].item())
        tok = id2word.get(idx, str(idx))
        grad_txt = _fmt_float(grad_pos) if grad_pos is not None else "n/a"
        lines.append(f"{tok}: p_topk={probs_topk[j].item():.4f} grad={grad_txt}")
    return lines


def _children(encoder, node) -> List[int]:
    ntype = node['type']
    idx = node['inorder_idx']
    if ntype == 'binary':
        left, right = encoder._get_binary_child_indices(idx)
        return [c for c in [left, right] if c is not None and c >= 0]
    if ntype == 'unary':
        c = encoder._get_unary_child_idx(idx)
        return [c] if c is not None and c >= 0 else []
    return []


def _layout_positions(encoder, width_scale: float = 1.8) -> Dict[int, Tuple[float, float]]:
    tree = encoder.tree
    pos: Dict[int, Tuple[float, float]] = {}

    root = tree.get_root_inorder_idx()

    def assign(node_idx: int, x_left: float, x_right: float):
        node = tree.get_node_by_inorder_idx(node_idx)
        if node is None:
            return x_left
        y = -float(node['layer'])
        if node['type'] == 'leaf':
            x = (x_left + x_right) / 2.0
            pos[node_idx] = (x, y)
            return x
        children = _children(encoder, node)
        if len(children) == 1:
            child_x = assign(children[0], x_left, x_right)
            pos[node_idx] = (child_x, y)
            return child_x
        if len(children) >= 2:
            mid = (x_left + x_right) / 2.0
            lx = assign(children[0], x_left, mid)
            rx = assign(children[1], mid, x_right)
            x = (lx + rx) / 2.0
            pos[node_idx] = (x, y)
            return x
        x = (x_left + x_right) / 2.0
        pos[node_idx] = (x, y)
        return x

    assign(root, 0.0, 1.0)
    width_scale = max(1.0, float(width_scale))
    for k, (x, y) in list(pos.items()):
        pos[k] = (((x - 0.5) * width_scale) + 0.5, y)
    return pos


def _draw_metadata_panel(
    *,
    ax_info,
    frame_meta: Optional[Dict[str, Any]],
    fex_env,
    logits: torch.Tensor,
    active_seq_len: int,
    block_set: Set[int],
    eval_points: int,
):
    if frame_meta is None:
        frame_meta = {}

    total_loss = frame_meta.get("loss_total")
    mse_loss = frame_meta.get("avg_mse")
    loss01 = frame_meta.get("loss01")
    subtree_depth = frame_meta.get("subtree_depth")
    inner_time_ms = frame_meta.get("inner_time_ms")
    active_positions = frame_meta.get("active_positions")
    subtree_root = frame_meta.get("subtree_root")
    block_text = "n/a"
    if isinstance(active_positions, list) and len(active_positions) > 0:
        preview = active_positions[:16]
        suffix = "..." if len(active_positions) > 16 else ""
        block_text = f"{preview}{suffix}"

    subtree_root_txt = "n/a"
    if subtree_root is not None:
        subtree_root_txt = str(subtree_root)
    elif isinstance(active_positions, list) and len(active_positions) > 0:
        subtree_root_txt = "from_positions"

    metrics_lines = [
        f"loss_total={_fmt_float(total_loss)}  mse={_fmt_float(mse_loss)}  loss01={_fmt_float(loss01)}",
        f"subtree_depth={subtree_depth if subtree_depth is not None else 'n/a'}  inner_step_time_ms={_fmt_float(inner_time_ms)}",
        f"active_subtree_root={subtree_root_txt}  active_positions={block_text}",
        "p_topk = softmax over displayed top-k logits; grad = mean abs gradient at this sequence position.",
    ]
    ax_info.text(
        0.01,
        0.98,
        "\n".join(metrics_lines),
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", alpha=0.9, linewidth=0.4),
    )

    gt_expr = frame_meta.get("gt_expr")
    hard_expr = _decode_hard_expression(fex_env, logits, active_seq_len)

    expr_lines = [
        f"whole-expression(hard top1): {hard_expr}",
        f"gt-expression: {gt_expr if gt_expr is not None else 'n/a'}",
    ]
    ax_info.text(
        0.01,
        0.68,
        "\n".join(expr_lines),
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    table = _build_eval_table(frame_meta, fex_env, logits, active_seq_len, eval_points)
    if table:
        cell_text, col_labels = table
        tbl = ax_info.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="lower left",
            bbox=[0.01, 0.02, 0.98, 0.46],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)


def _build_marked_expression(fex_env, logits: torch.Tensor, active_seq_len: int, block_set: Set[int]) -> str:
    id2word = getattr(fex_env, "equation_id2word", {})
    top1 = torch.argmax(logits[:active_seq_len], dim=-1)
    
    # Ensure top1 is always treatable as a tensor or has a length
    if not torch.is_tensor(top1):
        top1 = torch.tensor(top1)
        
    pieces: List[str] = []
    in_block = False
    for pos in range(1, max(1, active_seq_len - 1)):
        if pos >= top1.shape[0]:
            break
        if pos in block_set and len(block_set) > 0:
            if not in_block:
                pieces.append("(<current_tree>)")
                in_block = True
            continue
        in_block = False
        pieces.append(id2word.get(int(top1[pos].item()), "<UNK>"))
    return " ".join(pieces[:120])


def _decode_hard_expression(fex_env, logits: torch.Tensor, active_seq_len: int) -> str:
    try:
        top1 = torch.argmax(logits[:active_seq_len], dim=-1).detach().cpu().tolist()
        eos_id = fex_env.equation_word2id.get("<EOS>") if hasattr(fex_env, "equation_word2id") else None
        if eos_id is not None and len(top1) > 0 and top1[0] == eos_id:
            top1 = top1[1:]
        if eos_id is not None and len(top1) > 0 and top1[-1] == eos_id:
            top1 = top1[:-1]
        expr = fex_env.fex_encoder.decode(top1)
        return str(expr) if expr is not None else "<decode_failed>"
    except Exception:
        return "<decode_failed>"


def _build_eval_table(frame_meta, fex_env, logits, active_seq_len, eval_points):
    X_data = frame_meta.get("X_data")
    y_gt = frame_meta.get("Y_data")
    y_relaxed = frame_meta.get("y_relaxed")
    if not isinstance(X_data, torch.Tensor) or not isinstance(y_gt, torch.Tensor):
        return None

    X = X_data.detach().cpu()
    gt = y_gt.detach().cpu().view(-1)
    relaxed = y_relaxed.detach().cpu().view(-1) if isinstance(y_relaxed, torch.Tensor) else None
    hard = _eval_hard_prediction(fex_env, logits, active_seq_len, X)

    def get_size(obj):
        if torch.is_tensor(obj):
            return obj.numel()
        if isinstance(obj, np.ndarray):
            return obj.size
        return len(obj) if hasattr(obj, "__len__") else 0

    n = min(eval_points, X.size(0), get_size(gt))
    if relaxed is not None:
        n = min(n, get_size(relaxed))
    if hard is not None:
        n = min(n, get_size(hard))
    if n <= 0:
        return None

    rows = []
    for i in range(n):
        x_txt = _format_x_row(X[i])
        r = _fmt_float(relaxed[i].item() if hasattr(relaxed[i], "item") else relaxed[i]) if relaxed is not None else "n/a"
        h = _fmt_float(float(hard[i])) if hard is not None else "n/a"
        g = _fmt_float(gt[i].item() if hasattr(gt[i], "item") else gt[i])
        rows.append([str(i), x_txt, r, h, g])
    cols = ["idx", "x", "y_relaxed(subtree)", "y_whole(hard)", "y_gt"]
    return rows, cols


def _eval_hard_prediction(fex_env, logits, active_seq_len, X_tensor):
    try:
        top1 = torch.argmax(logits[:active_seq_len], dim=-1).detach().cpu().tolist()
        eos_id = fex_env.equation_word2id.get("<EOS>") if hasattr(fex_env, "equation_word2id") else None
        if eos_id is not None and len(top1) > 0 and top1[0] == eos_id:
            top1 = top1[1:]
        if eos_id is not None and len(top1) > 0 and top1[-1] == eos_id:
            top1 = top1[:-1]
        expr = fex_env.fex_encoder.decode(top1)
        if expr is None:
            return None
        tree = fex_env.simplifier.sympy_expr_to_tree(expr)
        fn = fex_env.simplifier.tree_to_numexpr_fn(tree)
        y_np = fn(X_tensor.numpy())
        if isinstance(y_np, np.ndarray) and y_np.ndim == 2 and y_np.shape[1] > 0:
            y_np = y_np[:, 0]
        return np.asarray(y_np).reshape(-1)
    except Exception:
        return None


def _format_x_row(x_row: torch.Tensor) -> str:
    vals = x_row.detach().cpu().view(-1).tolist()
    shown = vals[:3]
    text = ", ".join([f"{v:.3g}" for v in shown])
    if len(vals) > 3:
        text = text + ", ..."
    return f"[{text}]"


def _fmt_float(v) -> str:
    if v is None:
        return "n/a"
    try:
        x = float(v)
    except Exception:
        return "n/a"
    if math.isnan(x) or math.isinf(x):
        return str(x)
    return f"{x:.6g}"
