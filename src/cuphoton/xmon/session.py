# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Live collection and bounded interactive/automatic investigation."""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import os
import secrets
import threading
import uuid
from collections import deque
from collections.abc import Callable, Coroutine
from typing import Any

from .inference import InferenceConfig, Investigator
from .monitor import Monitor, now
from .nooa_backend import NooaError, _command


class MonitorSession:
    """Own the run's writer and investigation jobs until explicitly stopped.

    The poll callback reads application-defined completed inputs and publishes
    through the same monitor. Interactive and automatic queues share a bounded
    capacity; at most three interactive jobs precede a waiting automatic job.
    All state-writing work runs on this session's event loop. This is an
    in-process live session, not a distributed task executor.
    """

    def __init__(
        self,
        monitor: Monitor,
        *,
        config: InferenceConfig | None = None,
        instructions: str = "",
        poll: Callable[[], None] | None = None,
        poll_interval: float = 1.0,
        auto_investigate: bool = False,
        domain_query: Callable[[str, str], dict[str, Any]] | None = None,
        domain_views: dict[str, str] | None = None,
        access_token_env: str | None = None,
    ) -> None:
        if not 0.05 <= poll_interval <= 3600:
            raise ValueError(
                "poll interval must be between 0.05 and 3600 seconds"
            )
        if auto_investigate and config is None:
            raise ValueError(
                "automatic investigation requires inference configuration"
            )
        if len(instructions) > 8192:
            raise ValueError(
                "investigator instructions exceed 8192 characters"
            )
        self.monitor = monitor
        self.config = config
        self.instructions = instructions
        self.poll = poll
        self.poll_interval = poll_interval
        self.auto_investigate = auto_investigate
        self.domain_query = domain_query
        self.domain_views = domain_views or {}
        self._token = (
            os.environ.get(access_token_env, "") if access_token_env else ""
        )
        if access_token_env and not self._token:
            raise ValueError(
                "configured investigation access token is unavailable"
            )
        if self._token and len(self._token) < 24:
            raise ValueError(
                "investigation access token requires at least 24 characters"
            )
        self.reason = (
            "No inference configuration; monitoring remains available."
        )
        if config:
            self.reason = ""
            if importlib.util.find_spec("httpx") is None:
                self.reason = (
                    "Install cuphoton[xmon-inference] for investigation."
                )
            elif config.backend == "nooa":
                try:
                    _command()
                except NooaError as exc:
                    self.reason = str(exc)
        self._investigator = Investigator(config) if config else None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closing = False
        self._interactive: deque[dict[str, Any]] = deque()
        self._automatic: deque[dict[str, Any]] = deque()
        self._active: dict[str, asyncio.Task[None]] = {}
        self._seen: set[tuple[str, int]] = {
            (identifier, revision)
            for report in monitor.snapshot()["reports"]
            if report.get("trigger") == "automatic"
            for identifier, revision in report.get(
                "selected_evidence", report["evidence"]
            ).items()
        }

    @property
    def can_ask(self) -> bool:
        return self._investigator is not None and not self.reason

    @property
    def authentication_required(self) -> bool:
        return bool(self._token)

    def authenticate(self, supplied: str) -> bool:
        return not self._token or secrets.compare_digest(
            supplied, self._token
        )

    def ui(self) -> dict[str, Any]:
        return {
            "live": True,
            "can_ask": self.can_ask,
            "reason": self.reason,
            "authentication_required": self.authentication_required,
            "refresh_seconds": 1,
            "backend": self.config.backend if self.config else "",
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="xmon-session", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("monitor session did not start")

    def _run(self) -> None:
        asyncio.run(self._main())

    def _call(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        if self._loop is None or not self._loop.is_running() or self._closing:
            coroutine.close()
            raise ValueError("monitor session is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise ValueError(
                "monitor is busy collecting inputs; retry shortly"
            ) from exc

    def submit(
        self, question: str, sample_ids: list[str], *, scope: str = "selected"
    ) -> str:
        return str(
            self._call(self._submit(question, sample_ids, scope=scope))
        )

    async def _submit(
        self,
        question: str,
        sample_ids: list[str],
        *,
        scope: str,
        trigger: str = "question",
    ) -> str:
        if not self.can_ask:
            raise ValueError(self.reason)
        if (
            not isinstance(question, str)
            or not 1 <= len(question.strip()) <= 4096
        ):
            raise ValueError("question must contain 1–4096 characters")
        if scope not in {"selected", "run"} or not isinstance(
            sample_ids, list
        ):
            raise ValueError("invalid evidence scope")
        if any(not isinstance(identifier, str) for identifier in sample_ids):
            raise ValueError("sample IDs must be strings")
        maximum = self.config.max_samples if self.config else 64
        if (
            len(set(sample_ids)) != len(sample_ids)
            or len(sample_ids) > maximum
        ):
            raise ValueError(f"select at most {maximum} unique samples")
        if scope == "selected" and not sample_ids:
            raise ValueError("select a sample to investigate")
        state = self.monitor.snapshot()
        if not state["samples"] or any(
            identifier not in state["samples"] for identifier in sample_ids
        ):
            raise ValueError("selected evidence is not available")
        if (
            len(self._interactive) + len(self._automatic) + len(self._active)
            >= 16
        ):
            raise ValueError(
                "investigation capacity reached; retry after a job finishes"
            )
        identifier = uuid.uuid4().hex
        report: dict[str, Any] = {
            "id": identifier,
            "kind": "model_interpretation",
            "question": question.strip(),
            "scope": scope,
            "trigger": trigger,
            "status": "queued",
            "started_at": now(),
            "backend": self.config.backend if self.config else "",
            "events": [],
            "evidence": {
                key: state["samples"][key]["sample"]["revision"]
                for key in sample_ids
            },
        }
        report["selected_evidence"] = dict(report["evidence"])
        self.monitor.save_report(report)
        job = {"report": report, "sample_ids": list(sample_ids)}
        (
            self._automatic if trigger == "automatic" else self._interactive
        ).append(job)
        return identifier

    def cancel(self, report_id: str) -> bool:
        return bool(self._call(self._cancel(report_id)))

    async def _cancel(self, report_id: str) -> bool:
        task = self._active.get(report_id)
        if task:
            task.cancel()
            return True
        for queue in (self._interactive, self._automatic):
            for job in list(queue):
                if job["report"]["id"] == report_id:
                    queue.remove(job)
                    job["report"].update(
                        status="cancelled", finished_at=now()
                    )
                    self.monitor.save_report(job["report"])
                    return True
        return False

    async def _execute(self, job: dict[str, Any]) -> None:
        report = job["report"]
        try:
            current = self.monitor.snapshot()["samples"]
            if any(
                identifier not in current
                or current[identifier]["sample"]["revision"] != revision
                for identifier, revision in report[
                    "selected_evidence"
                ].items()
            ):
                report.update(status="stale", finished_at=now())
                self.monitor.save_report(report)
                return
            assert self._investigator is not None
            await self._investigator.investigate(
                self.monitor,
                job["sample_ids"],
                question=report["question"],
                instructions=self.instructions,
                scope=report["scope"],
                report_id=report["id"],
                trigger=report["trigger"],
                domain_query=self.domain_query,
                domain_views=self.domain_views,
            )
        except asyncio.CancelledError:
            current = next(
                (
                    item
                    for item in self.monitor.snapshot()["reports"]
                    if item["id"] == report["id"]
                ),
                report,
            )
            if current["status"] in {"queued", "running"}:
                current.update(status="cancelled", finished_at=now())
                self.monitor.save_report(current)
        except Exception as exc:
            report.update(
                status="failed",
                diagnostic=f"Investigation failed ({type(exc).__name__}).",
                finished_at=now(),
            )
            self.monitor.save_report(report)
        finally:
            self._active.pop(report["id"], None)

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._ready.set()
        next_poll = 0.0
        interactive_count = 0
        try:
            if self.poll:
                self.monitor._collector("running")
            while not self._closing:
                if self.poll and self._loop.time() >= next_poll:
                    try:
                        self.poll()
                        self.monitor._collector("watching")
                    except Exception as exc:
                        self.monitor._error("live-source", exc)
                        self.monitor._collector("failed")
                    next_poll = self._loop.time() + self.poll_interval
                stale, candidates = self.monitor._session_state(
                    include_candidates=self.auto_investigate and self.can_ask
                )
                for identifier in stale:
                    if identifier in self._active:
                        self._active[identifier].cancel()
                for identity in candidates:
                    if identity in self._seen:
                        continue
                    if (
                        len(self._interactive)
                        + len(self._automatic)
                        + len(self._active)
                        >= 16
                    ):
                        break
                    self._seen.add(identity)
                    try:
                        await self._submit(
                            "Investigate these findings and compare "
                            "relevant evidence in this run.",
                            [identity[0]],
                            scope="selected",
                            trigger="automatic",
                        )
                    except (ValueError, OSError) as exc:
                        self.auto_investigate = False
                        try:
                            self.monitor._error(
                                "auto-investigation-stopped", exc
                            )
                        except OSError:
                            # The monitor exposes failed persistence in its
                            # storage_error flag; collection can still retry.
                            pass
                        break
                capacity = self.config.max_concurrent if self.config else 1
                while len(self._active) < capacity and (
                    self._interactive or self._automatic
                ):
                    if self._interactive and (
                        interactive_count < 3 or not self._automatic
                    ):
                        job = self._interactive.popleft()
                        interactive_count += 1
                    else:
                        job = self._automatic.popleft()
                        interactive_count = 0
                    task = asyncio.create_task(self._execute(job))
                    self._active[job["report"]["id"]] = task
                await asyncio.sleep(0.05)
        except Exception as exc:
            self.reason = (
                "Live session stopped; review ingestion diagnostics."
            )
            self.monitor._error("live-session", exc)
            self.monitor._collector("failed")
        finally:
            for queue in (self._interactive, self._automatic):
                for job in queue:
                    job["report"].update(
                        status="interrupted", finished_at=now()
                    )
                    self.monitor.save_report(job["report"])
                queue.clear()
            tasks = list(self._active.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if self.poll and self.monitor.snapshot()["collector"] != "failed":
                self.monitor._collector("stopped")

    def close(self) -> None:
        self._closing = True
        if self._thread:
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                raise RuntimeError("monitor collection did not stop")
