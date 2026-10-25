"""Generate persistent demo sample images + ground-truth masks.

Writes PNGs into samples/ so the UI, README, and demo have stable files to
show. Deterministic (fixed seeds).
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.vision import synthetic  # noqa: E402

OUT = ROOT / "samples"
OUT.mkdir(parents=True, exist_ok=True)

PLAN = [
    ("normal", None, 101),
    ("scratch", "scratch", 202),
    ("spot", "spot", 303),
    ("dent", "dent", 404),
]


def main() -> None:
    for name, kind, seed in PLAN:
        s = synthetic.make_sample(seed, defect=kind)
        cv2.imwrite(str(OUT / f"{name}.png"), s.image)
        if kind is not None:
            cv2.imwrite(str(OUT / f"{name}_mask.png"), s.mask)
        print(f"wrote samples/{name}.png (label={s.label})")


if __name__ == "__main__":
    main()
