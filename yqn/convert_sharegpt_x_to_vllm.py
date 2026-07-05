from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".jsonl":
        return input_path.with_suffix(".vllm.json")
    return input_path.with_name(f"{input_path.name}.vllm.json")


def get_role(turn: dict) -> str:
    role = turn.get("role", {})
    if isinstance(role, dict):
        return role.get("role", "")
    if isinstance(role, str):
        return role
    return ""


def get_text(turn: dict) -> str:
    content = turn.get("content", {})
    raw = content.get("content", []) if isinstance(content, dict) else content

    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        return " ".join(part for part in raw if isinstance(part, str)).strip()
    return ""


def convert_item(item: dict) -> dict | None:
    input_parts: list[str] = []
    output_parts: list[str] = []

    for turn in item.get("conversations", []):
        if not isinstance(turn, dict):
            continue
        role = get_role(turn)
        kind = turn.get("kind", "")
        text = get_text(turn)
        if not text or kind != "text":
            continue
        if role == "user":
            input_parts.append(text)
        elif role == "assistant":
            output_parts.append(text)

    if not input_parts or not output_parts:
        return None

    return {
        "conversations": [
            {"from": "human", "value": " ".join(input_parts)},
            {"from": "gpt", "value": " ".join(output_parts)},
        ]
    }


def convert_file(input_path: Path, output_path: Path) -> int:
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    count = 0

    with input_path.open(encoding="utf-8") as fin, tmp_path.open(
        "w", encoding="utf-8"
    ) as fout:
        fout.write("[\n")
        first = True
        for line in fin:
            line = line.strip()
            if not line:
                continue
            converted = convert_item(json.loads(line))
            if converted is None:
                continue
            if not first:
                fout.write(",\n")
            json.dump(converted, fout, ensure_ascii=False)
            first = False
            count += 1
        fout.write("\n]\n")

    if count == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"no usable ShareGPT-X samples converted from {input_path}")

    os.replace(tmp_path, output_path)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT-X JSONL to vLLM benchmark ShareGPT JSON."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to ShareGPT-X ChatGPT-Simple.jsonl.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        help="Output JSON path. Defaults to <input>.vllm.json next to the input.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild output even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else default_output_path(input_path)
    )

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if output_path.is_file() and output_path.stat().st_size > 0 and not args.force:
        print(f"reuse existing converted dataset: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = convert_file(input_path, output_path)
    print(f"converted {count} samples: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
