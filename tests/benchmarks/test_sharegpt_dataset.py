# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.benchmarks.datasets import ShareGPTDataset


class _FakeTokenizer:
    def __call__(self, text: str, **kwargs) -> SimpleNamespace:
        tokens = text.split()
        return SimpleNamespace(input_ids=list(range(max(1, len(tokens)))))


@pytest.mark.benchmark
def test_sharegpt_dataset_loads_original_json(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sharegpt.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "What is vLLM exactly?"},
                        {"from": "gpt", "value": "An LLM serving engine."},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )

    dataset = ShareGPTDataset(
        dataset_path=str(dataset_path), disable_shuffle=True
    )
    samples = dataset.sample(
        tokenizer=_FakeTokenizer(),
        num_requests=1,
    )

    assert len(samples) == 1
    assert samples[0].prompt == "What is vLLM exactly?"
    assert samples[0].expected_output_len == 4


@pytest.mark.benchmark
def test_sharegpt_dataset_loads_sharegpt_x_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sharegpt_x.jsonl"
    record = {
        "conversations": [
            {
                "role": {"role": "user"},
                "kind": "text",
                "created": 1710000000,
                "content": {
                    "content": [
                        "Explain tensor parallelism",
                        "in one sentence.",
                    ]
                },
            },
            {
                "role": {"role": "assistant"},
                "kind": "text",
                "content": {"content": ["It shards model weights across GPUs."]},
            },
            {
                "role": {"role": "user"},
                "kind": "text",
                "content": {"content": ["Give me another example."]},
            },
        ]
    }
    dataset_path.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    dataset = ShareGPTDataset(
        dataset_path=str(dataset_path), disable_shuffle=True
    )
    samples = dataset.sample(
        tokenizer=_FakeTokenizer(),
        num_requests=1,
    )

    assert len(samples) == 1
    assert samples[0].prompt == "Explain tensor parallelism in one sentence."
    assert samples[0].expected_output_len == 6
