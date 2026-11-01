# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from io import StringIO

import pytest

from cuphoton.xbeam.job_text import (
    JobTextError,
    parse_job_text,
)


def test_parser_preserves_order_duplicates_and_empty_values():
    parsed = parse_job_text(
        StringIO("# heading\n A:  1 \nA: 2:3\nEMPTY:\n@ note\n")
    )
    assert parsed.get_all("A") == ("1", "2:3")
    assert parsed.get_unique("EMPTY") == ""
    assert [entry.line_number for entry in parsed.entries] == [2, 3, 4]
    assert list(parsed) == ["A", "EMPTY"]


@pytest.mark.parametrize("text", ["missing delimiter", ": value"])
def test_parser_rejects_malformed_noncomment_lines(text):
    with pytest.raises(JobTextError, match="<text>:1"):
        parse_job_text(StringIO(text))
