# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Streaming output integrity and numerical compatibility contracts."""

import hashlib


def execution_contract(fingerprint, profile, backend, autocorrelation_mode):
    return {
        "version": 1,
        "numerical_revision": 2,
        "job_fingerprint": fingerprint,
        "profile": profile,
        "backend": backend,
        "autocorrelation_mode": autocorrelation_mode,
    }


class HashingWriter:
    """Hash exactly the successfully written UTF-8 or binary bytes."""

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, value):
        written = self.stream.write(value)
        if written != len(value):
            raise OSError("short output write")
        data = value.encode("utf-8") if isinstance(value, str) else value
        self.digest.update(data)
        self.size += len(data)
        return written

    def summary(self):
        return {"size_bytes": self.size, "sha256": self.digest.hexdigest()}

    def close(self):
        self.stream.close()
