# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic collection with explicit recovery and revision semantics."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .records import Sample, encode, sample_from_dict


def now() -> str:
    return datetime.now(UTC).isoformat()


class Monitor:
    """Own one small run. Use one writer process per state directory.

    A new monitor refuses a nonempty directory. Recovery is explicit via
    ``load``. Snapshot replacement is atomic; a storage failure is raised and
    exposed in memory. This is a local run store, not a multiwriter database.
    """

    def __init__(self, directory: str | Path, *, run_id: str) -> None:
        if not isinstance(run_id, str) or not 1 <= len(run_id) <= 128:
            raise ValueError("run_id must contain 1–128 characters")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(
                "state exists; use Monitor.load or a new directory"
            )
        self._lock = threading.RLock()
        self._storage_error = False
        self._state: dict[str, Any] = {}
        self._commit(
            {
                "schema": 1,
                "run_id": run_id,
                "created_at": now(),
                "collector": "idle",
                "samples": {},
                "history": [],
                "errors": [],
                "reports": [],
            }
        )

    @classmethod
    def load(cls, directory: str | Path, *, recover: bool = True) -> Monitor:
        """Recover evidence, or read without mutation with ``recover=False``.

        Recovery requires ownership of the writer: unfinished collector and
        model work becomes interrupted. Viewers use ``recover=False``.
        """
        self = cls.__new__(cls)
        self.directory = Path(directory)
        self._lock = threading.RLock()
        self._storage_error = False
        state = json.loads((self.directory / "state.json").read_text())
        if state.get("schema") != 1:
            raise ValueError("unsupported monitor state schema")
        for key, record in state["samples"].items():
            sample = sample_from_dict(record["sample"])
            if key != sample.id or record["digest"] != self._digest(sample):
                raise ValueError("sample identity or digest mismatch")
        for record in state["history"]:
            if record["digest"] != self._digest(
                sample_from_dict(record["sample"])
            ):
                raise ValueError("history digest mismatch")
        changed = False
        for report in state["reports"]:
            if recover and report["status"] in {
                "pending",
                "queued",
                "running",
            }:
                report["status"] = "interrupted"
                report["finished_at"] = now()
                changed = True
        if recover and state["collector"] == "running":
            state["collector"] = "interrupted"
            changed = True
        self._state = state
        if changed:
            self._commit(state)
        return self

    def refresh(self) -> None:
        """Read the latest atomic snapshot without interrupting its writer."""
        with self._lock:
            loaded = Monitor.load(self.directory, recover=False)
            self._state = loaded._state

    @staticmethod
    def _digest(sample: Sample) -> str:
        return hashlib.sha256(encode(asdict(sample)).encode()).hexdigest()

    def _commit(self, state: dict[str, Any]) -> None:
        data = encode(state)
        temporary = self.directory / ".state.tmp"
        try:
            with temporary.open("w") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.directory / "state.json")
        except OSError:
            self._storage_error = True
            raise
        self._state = state
        self._storage_error = False

    def snapshot(self) -> dict[str, Any]:
        """Return detached state, including collector and report failures."""
        with self._lock:
            result = json.loads(encode(self._state))
            result["storage_error"] = self._storage_error
            return result

    @property
    def samples(self) -> list[Sample]:
        with self._lock:
            return sorted(
                (
                    sample_from_dict(x["sample"])
                    for x in self._state["samples"].values()
                ),
                key=lambda sample: (sample.sequence, sample.id),
            )

    def publish(self, sample: Sample) -> bool:
        """Publish new evidence; return False for an exact current duplicate.

        Changed content requires an increasing revision. Prior samples remain
        in history; reports using superseded evidence become visibly stale.
        Application aggregates should be recomputed from ``samples``.
        """
        sample = sample_from_dict(asdict(sample))
        digest = self._digest(sample)
        with self._lock:
            state = self.snapshot()
            state.pop("storage_error")
            old = state["samples"].get(sample.id)
            if old:
                if old["digest"] == digest:
                    return False
                if sample.revision <= old["sample"]["revision"]:
                    raise ValueError(
                        "changed evidence requires an increasing revision"
                    )
            if len(state["samples"]) >= 10000 and not old:
                raise ValueError(
                    "run reached 10000 samples; begin another run"
                )
            if len(state["history"]) >= 50000:
                raise ValueError(
                    "run reached 50000 revisions; begin another run"
                )
            record = {
                "sample": asdict(sample),
                "digest": digest,
                "processed_at": now(),
            }
            state["samples"][sample.id] = record
            state["history"].append(record)
            for report in state["reports"]:
                revisions = report.get("evidence", {})
                if (
                    sample.id in revisions
                    and revisions[sample.id] != sample.revision
                ):
                    report["status"] = "stale"
            self._commit(state)
            return True

    def _collector(self, status: str) -> None:
        with self._lock:
            if self._state["collector"] == status and not self._storage_error:
                return
            state = self.snapshot()
            state.pop("storage_error")
            state["collector"] = status
            self._commit(state)

    def _session_state(
        self, *, include_candidates: bool
    ) -> tuple[list[str], list[tuple[str, int]]]:
        """Read detached scheduling metadata without copying evidence."""
        with self._lock:
            stale = [
                report["id"]
                for report in self._state["reports"]
                if report["status"] == "stale"
            ]
            candidates = []
            if include_candidates:
                for record in self._state["samples"].values():
                    sample = record["sample"]
                    if any(
                        finding["severity"] in {"warning", "error"}
                        for finding in sample["findings"]
                    ):
                        candidates.append((sample["id"], sample["revision"]))
            return stale, candidates

    def _error(self, source_id: str, exc: Exception) -> None:
        # Exception text may contain credentials or local paths. Keep only
        # its class plus the explicit, application-owned source identifier.
        with self._lock:
            state = self.snapshot()
            state.pop("storage_error")
            state["errors"] = (
                state["errors"]
                + [
                    {
                        "source_id": source_id[:128],
                        "error_type": type(exc).__name__,
                        "category": "processing",
                        "processed_at": now(),
                    }
                ]
            )[-100:]
            self._commit(state)

    def ingest(
        self,
        item: Any,
        analyzer: Callable[[Any], Sample],
        *,
        source_id: str = "input",
    ) -> bool:
        """Analyze an item; persist a visible failure on analysis errors."""
        try:
            sample = analyzer(item)
        except Exception as exc:
            self._error(source_id, exc)
            return False
        try:
            return self.publish(sample)
        except (TypeError, ValueError) as exc:
            self._error(source_id, exc)
            return False

    def replay(
        self, items: Iterable[Any], analyzer: Callable[[Any], Sample]
    ) -> None:
        """Consume a source, exposing enumeration failure."""
        self._collector("running")
        try:
            for index, item in enumerate(items):
                self.ingest(item, analyzer, source_id=f"input-{index}")
        except Exception as exc:
            self._error("source", exc)
            self._collector("failed")
            raise
        self._collector("complete")

    def save_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """Store a separate report, invalidating it if evidence changed."""
        with self._lock:
            report = json.loads(encode(report))
            for identifier, revision in report["evidence"].items():
                current = self._state["samples"].get(identifier)
                if not current or current["sample"]["revision"] != revision:
                    report["status"] = "stale"
            if not self._storage_error and any(
                old == report for old in self._state["reports"]
            ):
                return report
            state = self.snapshot()
            state.pop("storage_error")
            reports = [x for x in state["reports"] if x["id"] != report["id"]]
            if len(reports) >= 1000:
                raise ValueError(
                    "run reached 1000 reports; begin another run"
                )
            state["reports"] = reports + [report]
            self._commit(state)
            return json.loads(encode(report))
