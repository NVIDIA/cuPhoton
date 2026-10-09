#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Run through make test-xdr-coverage with reentrant CFITSIO available.
set -euo pipefail
cd "$(dirname "$0")/../.."

bash scripts/wheels/install_build_dependencies.sh

coverage_build=build/xdr-coverage
rm -rf "$coverage_build"
mkdir -p "$coverage_build"

CUPHOTON_XDR_BUILD_EXT=1 CUPHOTON_XDR_COVERAGE=1 python setup.py \
    build_py --build-lib "$coverage_build/lib" \
    build_ext --force --build-temp "$coverage_build/obj" \
    --build-lib "$coverage_build/lib"

export PYTHONPATH="$PWD/$coverage_build/lib"
python - <<'PY'
from pathlib import Path
from cuphoton.xdr import nvcomp_batch

ext = nvcomp_batch._try_get_cpp_ext()
assert ext is not None, "Instrumented xDR extension did not load"
assert Path(ext.__file__).resolve().is_relative_to(
    Path("build/xdr-coverage/lib").resolve()
), ext.__file__
print(f"Testing instrumented extension: {ext.__file__}")
PY

python -m pytest -q -rs -o "pythonpath=$PYTHONPATH" \
    --junitxml="$coverage_build/junit.xml" \
    tests/xdr/test_native.py tests/xdr/test_deflate_gpu.py

python - <<'PY'
from xml.etree import ElementTree as ET

report = ET.parse("build/xdr-coverage/junit.xml")
assert report.findall(".//testcase"), "No native tests executed"
assert not report.findall(".//skipped"), "Native tests must not skip"
assert not report.findall(".//failure")
assert not report.findall(".//error")
PY

uv tool run --from gcovr==8.6 gcovr \
    --root . --filter 'src/cuphoton/xdr/src/' \
    --gcov-executable "${GCOV:-gcov}" \
    --xml "$coverage_build/coverage-native.xml" \
    --json "$coverage_build/coverage-native.json" \
    --print-summary "$coverage_build/obj"

python - <<'PY'
from pathlib import Path
from xml.etree import ElementTree as ET

report_path = Path("build/xdr-coverage/coverage-native.xml")
report = ET.parse(report_path)
# gcovr emits an absolute checkout root; make the report portable.
for source in report.findall("sources/source"):
    source.text = "."
measured = {
    item.attrib["filename"]
    for item in report.findall(".//class")
    if any(int(line.attrib["hits"]) > 0 for line in item.findall("lines/line"))
}
expected = {str(path) for path in Path("src/cuphoton/xdr/src").glob("*.cpp")}
assert expected and expected <= measured, (expected, measured)
report.write(report_path, encoding="utf-8", xml_declaration=True)
print("Measured execution in every xDR C++ translation unit")
PY

git rev-parse HEAD > "$coverage_build/coverage-revision.txt"
