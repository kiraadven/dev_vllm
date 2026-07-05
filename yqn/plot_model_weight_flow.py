#!/usr/bin/env python
"""Plot a forward-ordered weight-size diagram for transformer layers.

The script reads HuggingFace safetensors weights without materializing tensors,
groups per-layer tensors into forward-path components, and draws boxes whose
areas represent weight memory size. It is model-agnostic for common HF
transformer naming patterns and defaults to plotting the first three layers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from safetensors import safe_open

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}

COMPONENT_ORDER = [
    "input_norm",
    "attn_qkv",
    "attn_out",
    "post_norm",
    "router",
    "experts",
    "shared_expert",
    "mlp",
    "other",
]

COMPONENT_STYLE = {
    "input_norm": {
        "title": "Input norm",
        "stage": 0,
        "lane": 0.0,
        "color": "#8dd3c7",
    },
    "attn_qkv": {
        "title": "Attention Q/K/V",
        "stage": 1,
        "lane": 0.0,
        "color": "#80b1d3",
    },
    "attn_out": {
        "title": "Attention output",
        "stage": 2,
        "lane": 0.0,
        "color": "#377eb8",
    },
    "post_norm": {
        "title": "Post-attn norm",
        "stage": 3,
        "lane": 0.0,
        "color": "#b3de69",
    },
    "router": {
        "title": "Router / gate",
        "stage": 4,
        "lane": 0.0,
        "color": "#fdb462",
    },
    "experts": {
        "title": "Routed experts",
        "stage": 5,
        "lane": 0.34,
        "color": "#fb8072",
    },
    "shared_expert": {
        "title": "Shared expert",
        "stage": 5,
        "lane": -0.34,
        "color": "#bc80bd",
    },
    "mlp": {
        "title": "Dense MLP / FFN",
        "stage": 5,
        "lane": 0.0,
        "color": "#bebada",
    },
    "other": {
        "title": "Other layer weights",
        "stage": 5,
        "lane": -0.68,
        "color": "#d9d9d9",
    },
}

STAGE_LABELS = {
    0: "Norm",
    1: "Q/K/V proj",
    2: "Attn out",
    3: "Norm",
    4: "Router",
    5: "MoE / MLP",
    6: "Layer output",
}

DEFAULT_LAYER_RE = r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\."


@dataclass
class TensorInfo:
    name: str
    shard: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    layer_id: int | None


@dataclass
class Component:
    key: str
    layer_id: int
    nbytes: int = 0
    tensor_names: list[str] = field(default_factory=list)
    shapes: list[tuple[int, ...]] = field(default_factory=list)
    dtypes: set[str] = field(default_factory=set)

    @property
    def title(self) -> str:
        return COMPONENT_STYLE[self.key]["title"]

    @property
    def stage(self) -> int:
        return int(COMPONENT_STYLE[self.key]["stage"])

    @property
    def lane(self) -> float:
        return float(COMPONENT_STYLE[self.key]["lane"])

    @property
    def color(self) -> str:
        return str(COMPONENT_STYLE[self.key]["color"])


@dataclass
class DrawnBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> tuple[float, float]:
        return self.x - self.width / 2.0, self.y

    @property
    def right(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y


@dataclass
class LayerLayout:
    layer_id: int
    components: list[Component]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw per-layer model weight sizes as forward-ordered rectangles."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="HuggingFace model directory containing safetensors weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_weight_flow.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma/range layer list, e.g. '0,1,2' or '0-2'. Overrides --start-layer/--num-layers.",
    )
    parser.add_argument("--start-layer", type=int, default=0, help="First layer to plot.")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers to plot.")
    parser.add_argument(
        "--layer-regex",
        type=str,
        default=DEFAULT_LAYER_RE,
        help="Regex with one capture group for layer id. Default handles .layers.N, .h.N, .blocks.N.",
    )
    parser.add_argument(
        "--area-scale",
        choices=("linear", "sqrt", "log"),
        default="linear",
        help="How rectangle area is scaled from bytes. Linear is most faithful; log is most readable.",
    )
    parser.add_argument(
        "--min-box-frac",
        type=float,
        default=0.075,
        help="Minimum side fraction for tiny tensors so labels remain visible.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title. Default uses model directory name.",
    )
    return parser.parse_args()


def dtype_nbytes(dtype: str) -> int:
    normalized = dtype.upper().replace("TORCH.", "")
    if normalized in DTYPE_BYTES:
        return DTYPE_BYTES[normalized]
    match = re.search(r"(\d+)$", normalized)
    if match:
        return max(1, int(match.group(1)) // 8)
    raise ValueError(f"Unknown safetensors dtype: {dtype}")


def numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape) if shape else 1


def parse_layer_selection(args: argparse.Namespace) -> list[int] | None:
    if not args.layers:
        return None
    selected: list[int] = []
    for part in args.layers.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            selected.extend(range(int(start), int(end) + 1))
        else:
            selected.append(int(token))
    return sorted(dict.fromkeys(selected))


def load_safetensor_weights(model_dir: Path, layer_regex: str) -> list[TensorInfo]:
    index_path = model_dir / "model.safetensors.index.json"
    shard_to_names: dict[str, list[str]] = defaultdict(list)
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        for name, shard in weight_map.items():
            shard_to_names[str(shard)].append(str(name))
    else:
        shards = sorted(model_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No safetensors weights found in {model_dir}")
        for shard_path in shards:
            shard_to_names[shard_path.name] = []

    layer_re = re.compile(layer_regex)
    tensors: list[TensorInfo] = []
    for shard_name, requested_names in sorted(shard_to_names.items()):
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard listed in index does not exist: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            names = requested_names or list(handle.keys())
            for name in names:
                weight_slice = handle.get_slice(name)
                dtype = str(weight_slice.get_dtype())
                shape = tuple(int(dim) for dim in weight_slice.get_shape())
                layer_match = layer_re.search(name)
                layer_id = int(layer_match.group(1)) if layer_match else None
                tensors.append(
                    TensorInfo(
                        name=name,
                        shard=shard_name,
                        shape=shape,
                        dtype=dtype,
                        nbytes=numel(shape) * dtype_nbytes(dtype),
                        layer_id=layer_id,
                    )
                )
    return tensors


def classify_tensor(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("input_layernorm", "input_norm", ".ln_1.", ".ln1.")):
        return "input_norm"
    if any(
        token in lowered
        for token in (
            "post_attention_layernorm",
            "post_attn_layernorm",
            "post_attention_norm",
            ".ln_2.",
            ".ln2.",
        )
    ):
        return "post_norm"
    if is_attention_tensor(lowered):
        if any(
            token in lowered
            for token in (
                "q_proj",
                "k_proj",
                "v_proj",
                "query",
                "key",
                "value",
                "c_attn",
                "wq",
                "wk",
                "wv",
            )
        ):
            return "attn_qkv"
        return "attn_out"
    if is_expert_tensor(lowered):
        return "experts"
    if "shared_expert" in lowered or "shared_experts" in lowered:
        return "shared_expert"
    if is_router_tensor(lowered):
        return "router"
    if any(token in lowered for token in (".mlp.", ".feed_forward.", ".ffn.", ".mlp_")):
        return "mlp"
    return "other"


def is_attention_tensor(lowered: str) -> bool:
    return any(
        token in lowered
        for token in (
            "self_attn",
            ".attention.",
            ".attn.",
            ".attn_",
            ".mha.",
        )
    )


def is_expert_tensor(lowered: str) -> bool:
    return any(
        token in lowered
        for token in (
            ".experts.",
            ".expert.",
            "block_sparse_moe.experts",
            "moe.experts",
        )
    ) and "shared_expert" not in lowered


def is_router_tensor(lowered: str) -> bool:
    if "gate_proj" in lowered or ".experts." in lowered:
        return False
    return any(token in lowered for token in (".router.", ".gate.weight", ".gate.bias"))


def build_layer_layouts(tensors: list[TensorInfo], selected_layers: list[int]) -> list[LayerLayout]:
    by_layer: dict[int, dict[str, Component]] = defaultdict(dict)
    selected = set(selected_layers)
    for tensor in tensors:
        if tensor.layer_id not in selected:
            continue
        layer_id = int(tensor.layer_id)
        key = classify_tensor(tensor.name)
        component = by_layer[layer_id].setdefault(key, Component(key=key, layer_id=layer_id))
        component.nbytes += tensor.nbytes
        component.tensor_names.append(tensor.name)
        component.shapes.append(tensor.shape)
        component.dtypes.add(tensor.dtype)

    layouts: list[LayerLayout] = []
    for layer_id in selected_layers:
        components = [by_layer[layer_id][key] for key in COMPONENT_ORDER if key in by_layer[layer_id]]
        if components:
            layouts.append(LayerLayout(layer_id=layer_id, components=components))
    return layouts


def choose_layers(tensors: list[TensorInfo], args: argparse.Namespace) -> list[int]:
    explicit = parse_layer_selection(args)
    available = sorted({tensor.layer_id for tensor in tensors if tensor.layer_id is not None})
    if not available:
        raise ValueError("No transformer layer ids were found. Try adjusting --layer-regex.")
    if explicit is not None:
        return [layer_id for layer_id in explicit if layer_id in available]
    return [
        layer_id
        for layer_id in available
        if args.start_layer <= layer_id < args.start_layer + args.num_layers
    ]


def scale_area(nbytes: int, mode: str) -> float:
    if mode == "linear":
        return float(nbytes)
    if mode == "sqrt":
        return math.sqrt(float(nbytes))
    if mode == "log":
        return math.log2(float(nbytes) + 1.0)
    raise ValueError(f"Unsupported area scale: {mode}")


def plot_weight_flow(
    layouts: list[LayerLayout],
    model_dir: Path,
    output: Path,
    area_scale: str,
    min_box_frac: float,
    title: str | None,
    dpi: int,
) -> None:
    if not layouts:
        raise ValueError("No layer components to plot")

    components = [component for layout in layouts for component in layout.components]
    max_scaled = max(scale_area(component.nbytes, area_scale) for component in components)
    stage_gap = 2.75
    x_by_stage = {stage: stage * stage_gap for stage in range(7)}
    row_gap = 2.45
    max_box_w = 1.42
    max_box_h = 0.86
    min_w = max_box_w * min_box_frac
    min_h = max_box_h * min_box_frac

    fig_w = 19.5
    fig_h = max(5.8, 2.7 + row_gap * len(layouts))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    y_by_layer = {
        layout.layer_id: (len(layouts) - index - 1) * row_gap for index, layout in enumerate(layouts)
    }
    drawn: dict[tuple[int, str], DrawnBox] = {}
    output_points: dict[int, tuple[float, float]] = {}

    header_y = max(y_by_layer.values()) + 1.42
    for stage, label in STAGE_LABELS.items():
        ax.text(
            x_by_stage[stage],
            header_y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )

    for layout in layouts:
        y_base = y_by_layer[layout.layer_id]
        ax.text(
            -1.0,
            y_base,
            f"Layer {layout.layer_id}",
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#333333",
        )
        ax.hlines(
            y_base - 0.98,
            xmin=-0.75,
            xmax=x_by_stage[6] + 0.9,
            color="#eeeeee",
            linewidth=1.0,
            zorder=0,
        )

        for component in layout.components:
            scaled = scale_area(component.nbytes, area_scale)
            side = math.sqrt(scaled / max_scaled) if max_scaled > 0 else 1.0
            width = max(min_w, max_box_w * side)
            height = max(min_h, max_box_h * side)
            x = x_by_stage[component.stage]
            y = y_base + component.lane
            rect = Rectangle(
                (x - width / 2.0, y - height / 2.0),
                width,
                height,
                facecolor=component.color,
                edgecolor="#333333",
                linewidth=0.8,
                alpha=0.86,
                zorder=3,
            )
            ax.add_patch(rect)
            drawn[(layout.layer_id, component.key)] = DrawnBox(x=x, y=y, width=width, height=height)
            ax.text(
                x + width / 2.0 + 0.08,
                y,
                component_label(component),
                ha="left",
                va="center",
                fontsize=7.2,
                color="#222222",
                linespacing=1.12,
                zorder=4,
            )

        output_x = x_by_stage[6]
        output_y = y_base
        output_points[layout.layer_id] = (output_x, output_y)
        ax.scatter([output_x], [output_y], s=42, color="#555555", zorder=4)
        ax.text(
            output_x + 0.14,
            output_y,
            "output\n(no weight)",
            ha="left",
            va="center",
            fontsize=7.2,
            color="#555555",
        )
        draw_layer_arrows(ax, layout, drawn, output_points[layout.layer_id])

    right_margin_x = x_by_stage[6] + 0.65
    for prev, nxt in zip(layouts[:-1], layouts[1:], strict=True):
        start = output_points[prev.layer_id]
        next_input = first_component_anchor(nxt, drawn, side="left")
        if next_input is not None:
            add_elbow_arrow(
                ax,
                start,
                next_input,
                bend_x=right_margin_x,
                color="#666666",
                linestyle="--",
            )

    legend_handles = [
        Patch(facecolor=COMPONENT_STYLE[key]["color"], edgecolor="#333333", label=COMPONENT_STYLE[key]["title"])
        for key in COMPONENT_ORDER
        if any(component.key == key for component in components)
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(5, len(legend_handles)),
        frameon=True,
        fontsize=8.5,
    )

    scale_note = {
        "linear": "Box area is linear in parameter bytes; very small tensors use a minimum visible size.",
        "sqrt": "Box area is scaled by sqrt(parameter bytes) for readability; labels show actual bytes.",
        "log": "Box area is scaled by log2(parameter bytes) for readability; labels show actual bytes.",
    }[area_scale]
    ax.text(
        0.0,
        -1.35,
        scale_note + " Arrows show forward data dependency order, not tensor size.",
        ha="left",
        va="center",
        fontsize=9,
        color="#555555",
    )

    plot_title = title or f"{model_dir.name}: weight-size flow for layers {layer_span(layouts)}"
    ax.set_title(plot_title, fontsize=15, fontweight="bold", pad=18)
    ax.set_xlim(-1.3, x_by_stage[6] + 2.2)
    ax.set_ylim(-1.65, header_y + 0.55)
    ax.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def component_label(component: Component) -> str:
    return (
        f"{component.title}\n"
        f"{format_bytes(component.nbytes)} · {len(component.tensor_names)} tensors\n"
        f"{short_name_examples(component.tensor_names)}"
    )


def short_name_examples(names: list[str]) -> str:
    if not names:
        return ""
    simplified = [simplify_tensor_name(name) for name in names]
    if len(simplified) == 1:
        return simplified[0]
    if len(simplified) == 2:
        return " + ".join(simplified)
    return f"{simplified[0]} + {simplified[1]} + ..."


def simplify_tensor_name(name: str) -> str:
    name = re.sub(r"^model\.layers\.\d+\.", "", name)
    name = re.sub(r"^transformer\.h\.\d+\.", "", name)
    name = re.sub(r"^layers\.\d+\.", "", name)
    if len(name) <= 42:
        return name
    return "..." + name[-39:]


def draw_layer_arrows(
    ax: plt.Axes,
    layout: LayerLayout,
    drawn: dict[tuple[int, str], DrawnBox],
    output_point: tuple[float, float],
) -> None:
    layer_id = layout.layer_id
    present = {component.key for component in layout.components}
    main_order = [key for key in ["input_norm", "attn_qkv", "attn_out", "post_norm", "router"] if key in present]
    for left_key, right_key in zip(main_order[:-1], main_order[1:], strict=True):
        add_arrow(ax, drawn[(layer_id, left_key)].right, drawn[(layer_id, right_key)].left)

    branch_keys = [key for key in ["experts", "shared_expert", "mlp", "other"] if key in present]
    if branch_keys:
        source_key = "router" if "router" in present else (main_order[-1] if main_order else None)
        if source_key is not None:
            for branch_key in branch_keys:
                add_arrow(ax, drawn[(layer_id, source_key)].right, drawn[(layer_id, branch_key)].left)
        for branch_key in branch_keys:
            add_arrow(ax, drawn[(layer_id, branch_key)].right, output_point)
    elif main_order:
        add_arrow(ax, drawn[(layer_id, main_order[-1])].right, output_point)


def first_component_anchor(
    layout: LayerLayout,
    drawn: dict[tuple[int, str], DrawnBox],
    side: str,
) -> tuple[float, float] | None:
    for key in COMPONENT_ORDER:
        box = drawn.get((layout.layer_id, key))
        if box is None:
            continue
        return box.left if side == "left" else box.right
    return None


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#444444",
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.85,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=2,
    )
    ax.add_patch(arrow)


def add_elbow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    bend_x: float,
    color: str = "#444444",
    linestyle: str = "-",
) -> None:
    sx, sy = start
    ex, ey = end
    mid_y = (sy + ey) / 2.0
    ax.plot(
        [sx, bend_x, bend_x, ex],
        [sy, sy, mid_y, mid_y],
        color=color,
        linestyle=linestyle,
        linewidth=0.85,
        zorder=1,
    )
    ax.plot(
        [ex, ex],
        [mid_y, ey],
        color=color,
        linestyle=linestyle,
        linewidth=0.85,
        zorder=1,
    )
    arrow = FancyArrowPatch(
        (ex, mid_y),
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.85,
        color=color,
        linestyle=linestyle,
        shrinkA=0,
        shrinkB=2,
        zorder=2,
    )
    ax.add_patch(arrow)


def format_bytes(nbytes: int) -> str:
    value = float(nbytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{value:.0f} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def layer_span(layouts: list[LayerLayout]) -> str:
    layers = [layout.layer_id for layout in layouts]
    if len(layers) == 1:
        return str(layers[0])
    if layers == list(range(min(layers), max(layers) + 1)):
        return f"{min(layers)}-{max(layers)}"
    return ",".join(str(layer) for layer in layers)


def print_summary(layouts: list[LayerLayout], output: Path) -> None:
    print(f"Wrote figure: {output}")
    for layout in layouts:
        total = sum(component.nbytes for component in layout.components)
        print(f"Layer {layout.layer_id}: {format_bytes(total)}")
        for component in layout.components:
            print(
                f"  - {component.title:<20} {format_bytes(component.nbytes):>12} "
                f"({len(component.tensor_names)} tensors)"
            )


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    tensors = load_safetensor_weights(model_dir, args.layer_regex)
    selected_layers = choose_layers(tensors, args)
    if not selected_layers:
        raise ValueError("Selected layers are empty; check --layers or --start-layer/--num-layers.")
    layouts = build_layer_layouts(tensors, selected_layers)
    plot_weight_flow(
        layouts=layouts,
        model_dir=model_dir,
        output=output,
        area_scale=args.area_scale,
        min_box_frac=args.min_box_frac,
        title=args.title,
        dpi=args.dpi,
    )
    print_summary(layouts, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
