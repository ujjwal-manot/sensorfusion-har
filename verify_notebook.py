"""Cell-by-cell verifier for the optimized notebook.

For each code cell:
1. Compile the source to bytecode (catches syntax errors).
2. Run a static analysis pass for common bugs:
   - Tuple-unpacking from MSM (must be `recon, mask = msm(...)`)
   - Tensor dtype safety (every numpy from augmentor cast to float32)
   - Magic strings that should match repo (REPO_URL, attribute names)
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

NB_PATH = Path(__file__).parent / "sensorfusion_har_OPTIMIZED.ipynb"

with NB_PATH.open(encoding="utf-8") as f:
    nb = json.load(f)

errors = []
warnings = []

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if not src.strip():
        continue

    cleaned_lines = []
    in_magic = False
    for line in src.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            in_magic = True
            cleaned_lines.append("# " + line)
            continue
        if in_magic and (line.rstrip().endswith("\\") or stripped == "" or line.startswith(" ")):
            cleaned_lines.append("# " + line)
            if not line.rstrip().endswith("\\"):
                in_magic = False
            continue
        in_magic = False
        cleaned_lines.append(line)
    cleaned = "".join(cleaned_lines)

    try:
        ast.parse(cleaned)
    except SyntaxError as e:
        errors.append(f"cell {idx}: SyntaxError: {e}")
        continue

    if "msm(" in src and "x, mask=mask" in src and "recon, ret_mask = msm" not in src and "recon, mask = msm" not in src:
        if "msm(x" in src:
            warnings.append(f"cell {idx}: looks like MSM call without tuple unpack")

    if "augmentor(" in src and "torch.from_numpy" in src:
        if "astype(np.float32)" not in src and "np.float32" not in src:
            warnings.append(f"cell {idx}: numpy->torch path without explicit float32 cast")

    if "REPO_URL" in src and "ujjwal-manot" not in src and "REPO_URL = " in src:
        warnings.append(f"cell {idx}: REPO_URL should point to ujjwal-manot fork")

print(f"checked {len(nb['cells'])} cells ({sum(1 for c in nb['cells'] if c['cell_type']=='code')} code)")
if errors:
    print("\nERRORS:")
    for e in errors:
        print(f"  {e}")
if warnings:
    print("\nWARNINGS:")
    for w in warnings:
        print(f"  {w}")
if not errors and not warnings:
    print("\nALL CELLS CLEAN")

sys.exit(1 if errors else 0)
