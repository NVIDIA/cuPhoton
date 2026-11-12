# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Authenticated GitHub operations and exact workflow-run tracking."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

from cuphoton_dev.build import FINAL_TAG

REPOSITORY = "NVIDIA/cuPhoton"
SITE = "https://nvidia.github.io/cuPhoton/"


class PublicationError(RuntimeError):
    """Publication failure retaining the known request and workflow state."""

    def __init__(self, message: str, result: dict[str, Any]) -> None:
        super().__init__(message)
        self.result = {**result, "error": message}


def api(
    path: str, *, data: dict[str, Any] | None = None, missing_ok: bool = False
) -> Any:
    args = ["gh", "api", "-H", "X-GitHub-Api-Version: 2026-03-10", path]
    if data is not None:
        args.extend(["--method", "POST", "--input", "-"])
    result = subprocess.run(
        args,
        input=json.dumps(data) if data is not None else None,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    if result.returncode:
        if missing_ok and "HTTP 404" in result.stderr:
            return None
        raise RuntimeError(f"GitHub request failed: {result.stderr.strip()}")
    return json.loads(result.stdout) if result.stdout.strip() else None


def releases(repository: str = REPOSITORY) -> list[dict[str, Any]]:
    result = []
    page = 1
    while True:
        batch = api(f"repos/{repository}/releases?per_page=100&page={page}")
        result.extend(batch)
        if len(batch) < 100:
            return result
        page += 1


def eligible(release: dict[str, Any]) -> bool:
    return bool(
        FINAL_TAG.fullmatch(release["tag_name"])
        and not release["draft"]
        and not release["prerelease"]
        and release["published_at"]
    )


def has_guide(tag: str, repository: str = REPOSITORY) -> bool:
    return (
        api(
            f"repos/{repository}/contents/docs/guide/_quarto.yml"
            f"?ref={quote(tag, safe='')}",
            missing_ok=True,
        )
        is not None
    )


def published_catalog() -> dict[str, Any]:
    try:
        with urlopen(SITE + "versions.json", timeout=30) as response:
            catalog = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return {"schema": 1, "stable": None, "versions": []}
        raise
    if catalog.get("schema") != 1 or not isinstance(
        catalog.get("versions"), list
    ):
        raise ValueError(
            "Published version catalog has an unsupported schema"
        )
    return dict(catalog)


def require_repository(repository: str) -> None:
    if repository != REPOSITORY:
        raise ValueError(f"Documentation publication supports {REPOSITORY}")


def docs_status(*, repository: str = REPOSITORY) -> dict[str, Any]:
    require_repository(repository)
    catalog = published_catalog()
    published = {entry["version"] for entry in catalog["versions"]}
    entries = []
    for release in releases(repository):
        tag = release["tag_name"]
        if not eligible(release):
            state = "ineligible"
        elif tag in published:
            state = "published"
        else:
            state = "missing" if has_guide(tag, repository) else "unavailable"
        entries.append(
            {
                "tag": tag,
                "state": state,
                "draft": release["draft"],
                "prerelease": release["prerelease"],
                "release_url": release["html_url"],
            }
        )
    return {
        "repository": repository,
        "site": SITE,
        "stable": catalog["stable"],
        "main": "published" if "main" in published else "missing",
        "releases": entries,
    }


def wait_for_run(
    repository: str, run_id: int, *, wait: bool, request_id: str
) -> dict[str, Any]:
    deadline = time.monotonic() + 7200
    result: dict[str, Any] = {
        "request_id": request_id,
        "run_id": run_id,
        "url": None,
        "status": "unknown",
        "conclusion": None,
        "controller_sha": None,
    }
    while True:
        try:
            run = api(f"repos/{repository}/actions/runs/{run_id}")
        except (RuntimeError, OSError, ValueError) as exc:
            raise PublicationError(
                f"Cannot read documentation publication run {run_id}: {exc}",
                result,
            ) from exc
        result = {
            "request_id": request_id,
            "run_id": run_id,
            "url": run["html_url"],
            "status": run["status"],
            "conclusion": run["conclusion"],
            "controller_sha": run["head_sha"],
        }
        if run["status"] == "completed":
            if run["conclusion"] != "success":
                raise PublicationError(
                    f"Documentation publication {run['conclusion']}: "
                    f"{run['html_url']}",
                    result,
                )
            return result
        if not wait:
            return result
        if time.monotonic() >= deadline:
            raise PublicationError(
                f"Publication still running; inspect {run['html_url']}",
                result,
            )
        time.sleep(5)


def docs_publish(
    *,
    ref: str | None = None,
    missing: bool = False,
    wait: bool = True,
    repository: str = REPOSITORY,
) -> dict[str, Any]:
    require_repository(repository)
    if missing == (ref is not None):
        raise ValueError("Specify exactly one of --ref or --missing")
    tag = "missing" if missing else ref
    if tag not in {"main", "missing"}:
        if not tag or not FINAL_TAG.fullmatch(tag):
            raise ValueError(
                "Publication ref must be main or a full release tag vX.Y.Z"
            )
        release = next(
            (r for r in releases(repository) if r["tag_name"] == tag), None
        )
        if release is None or not eligible(release):
            raise ValueError(f"{tag} is not a published final GitHub Release")
        if not has_guide(tag, repository):
            raise ValueError(f"Documentation is unavailable at {tag}")
    request_id = uuid.uuid4().hex
    try:
        response = api(
            f"repos/{repository}/actions/workflows/docs.yml/dispatches",
            data={
                "ref": "main",
                "inputs": {"tag": tag, "request_id": request_id},
            },
        )
    except (RuntimeError, OSError, ValueError) as exc:
        raise PublicationError(
            f"Dispatch outcome unknown for request {request_id}: {exc}. "
            "Inspect that request before redispatching.",
            {
                "request_id": request_id,
                "run_id": None,
                "url": None,
                "status": "dispatch_unknown",
                "conclusion": None,
                "controller_sha": None,
            },
        ) from exc
    run_id = response.get("workflow_run_id") if response else None
    if run_id is None:
        # Older API deployments return 204. Correlate our unique request
        # instead of selecting another caller's latest run.
        accepted = {
            "request_id": request_id,
            "run_id": None,
            "url": None,
            "status": "dispatch_accepted",
            "conclusion": None,
            "controller_sha": None,
        }
        deadline = time.monotonic() + 180
        while run_id is None:
            try:
                runs = api(
                    f"repos/{repository}/actions/workflows/docs.yml/runs"
                    "?event=workflow_dispatch&branch=main&per_page=100"
                )
            except (RuntimeError, OSError, ValueError) as exc:
                raise PublicationError(
                    f"Dispatch accepted; cannot find request {request_id}: "
                    f"{exc}. Do not redispatch blindly.",
                    accepted,
                ) from exc
            matches = [
                r
                for r in runs["workflow_runs"]
                if request_id in r["display_title"]
            ]
            if len(matches) > 1:
                raise PublicationError(
                    f"Ambiguous workflow request {request_id}", accepted
                )
            if matches:
                run_id = matches[0]["id"]
            elif time.monotonic() >= deadline:
                raise PublicationError(
                    f"Dispatch accepted; run not found for {request_id}. "
                    "Do not redispatch blindly.",
                    accepted,
                )
            else:
                time.sleep(3)
    return wait_for_run(
        repository, int(run_id), wait=wait, request_id=request_id
    )
