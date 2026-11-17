# SPDX-License-Identifier: Apache-2.0
"""Extract owned CUDA sources without importing runtime modules."""

import ast
import sys
from pathlib import Path

root, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
for filename, symbol, destination in [
    ("cuda_front.py", "_FRONT_SOURCE", "front.inc"),
    ("cuda_channelize.py", "_CHANNELIZE_RESIDUAL_SOURCE", "channel.inc"),
    ("core/cuda_products.py", "_VISIBILITY_SOURCE", "products.inc"),
    (
        "core/cuda_products.py",
        "_MULTI_VISIBILITY_SOURCE",
        "multi_products.inc",
    ),
]:
    tree = ast.parse((root / filename).read_text())
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == symbol for t in node.targets
        )
    ]
    if len(values) != 1:
        raise RuntimeError(f"expected one {symbol} definition")
    (output / destination).write_text(
        "// SPDX-License-Identifier: Apache-2.0\n" + values[0]
    )
