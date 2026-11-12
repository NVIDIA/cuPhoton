// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

import { readdirSync, readFileSync } from "node:fs";
import { register } from "node:module";
import { fileURLToPath } from "node:url";

register(new URL("./json-loader.mjs", import.meta.url));
const {
  convertWorkflowTemplate,
  isMapping,
  NoOperationTraceWriter,
  parseWorkflow,
} = await import("@actions/workflow-parser");

const directory = fileURLToPath(
  new URL("../../.github/workflows/", import.meta.url),
);
const files = process.argv.length > 2
  ? process.argv.slice(2)
  : readdirSync(directory)
      .filter((name) => /\.ya?ml$/.test(name))
      .sort()
      .map((name) => `${directory}/${name}`);

if (files.length === 0) {
  throw new Error("No GitHub Actions workflows found");
}

let failures = 0;
for (const file of files) {
  try {
    const result = parseWorkflow(
      { name: file, content: readFileSync(file, "utf8") },
      new NoOperationTraceWriter(),
    );
    if (result.value) {
      const workflow = await convertWorkflowTemplate(
        result.context, result.value,
      );
      // The published schema accepts these values independently; GitHub
      // also rejects this combination for workflow and job concurrency.
      const groups = [
        workflow.concurrency,
        ...(workflow.jobs ?? []).map((job) => job.concurrency),
      ];
      for (const group of groups) {
        if (
          isMapping(group) && group.find("queue")?.value === "max" &&
          group.find("cancel-in-progress")?.value === true
        ) {
          result.context.error(
            group,
            "queue: max cannot be combined with cancel-in-progress: true",
          );
        }
      }
    }
    const errors = result.context.errors.getErrors();
    for (const error of errors) {
      console.error(error.message);
    }
    failures += errors.length;
  } catch (error) {
    console.error(`${file}: ${error.message}`);
    failures += 1;
  }
}

console.log(`Validated ${files.length} workflow files; ${failures} errors.`);
process.exitCode = failures ? 1 : 0;
