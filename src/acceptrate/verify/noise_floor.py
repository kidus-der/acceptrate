"""Calibrate the cross-kernel noise floor of a backend.

Batched verification (N tokens per forward) and sequential decoding (N = 1)
take different kernel paths. On Metal fp16 their logits for the same
position disagree by ~0.03 typically and ~0.10 at worst
(docs/gates/P2.md). Below that margin, "the token plain decoding would
produce" is undefined, so the losslessness gate classifies a divergence at
such a position as a near-tie rather than a failure.

Measured through the Backend protocol only: no framework imports here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from acceptrate.backend.protocol import Backend

DEFAULT_CALIBRATION_PATH = Path("calibration/noise_floor.json")


@dataclass(frozen=True)
class NoiseFloor:
    chip: str
    mlx_version: str
    positions: int
    p50: float
    p99: float
    max: float
    argmax_flips: int

    @property
    def threshold(self) -> float:
        """A divergence at a sequential top-2 margin <= this is a near-tie."""
        return self.max


def is_near_tie(margin: float, floor: NoiseFloor) -> bool:
    return margin <= floor.threshold


def measure_noise_floor(
    backend: Backend,
    prompts: Sequence[Sequence[int]],
    gen_tokens: int,
    window: int,
    chip: str = "unknown",
    mlx_version: str = "unknown",
) -> NoiseFloor:
    """Greedy-decode sequentially, then re-score the same tokens in batched windows.

    For every generated position the sequential logits row and the batched
    row are compared: max |delta| and whether the argmax flips.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    deltas: list[float] = []
    flips = 0
    for prompt in prompts:
        logits = backend.prefill(prompt)
        generated: list[int] = []
        sequential_rows: list[np.ndarray] = []
        for _ in range(gen_tokens):
            token = int(np.argmax(logits))
            generated.append(token)
            logits = backend.decode_step(token)
            sequential_rows.append(logits)
        backend.prefill(prompt)
        for start in range(0, gen_tokens, window):
            chunk = generated[start : start + window]
            batched_rows = backend.verify(chunk)
            for j in range(len(chunk)):
                seq_row = sequential_rows[start + j]
                deltas.append(float(np.max(np.abs(batched_rows[j] - seq_row))))
                flips += int(np.argmax(batched_rows[j]) != np.argmax(seq_row))
    arr = np.array(deltas, dtype=np.float64)
    return NoiseFloor(
        chip=chip,
        mlx_version=mlx_version,
        positions=len(deltas),
        p50=float(np.median(arr)),
        p99=float(np.percentile(arr, 99)),
        max=float(arr.max()),
        argmax_flips=flips,
    )


def save_noise_floor(floor: NoiseFloor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(floor), indent=2) + "\n")


def load_noise_floor(path: Path) -> NoiseFloor:
    data = json.loads(path.read_text())
    missing = [f for f in NoiseFloor.__dataclass_fields__ if f not in data]
    if missing:
        raise ValueError(f"{path} is not a noise-floor calibration: missing {missing}")
    return NoiseFloor(**{k: data[k] for k in NoiseFloor.__dataclass_fields__})
