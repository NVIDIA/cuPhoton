// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

// The parser's published ESM omits Node's required JSON import attribute.
export async function load(url, context, nextLoad) {
  if (url.endsWith("/@actions/workflow-parser/dist/workflow-v1.0.min.json")) {
    context = { ...context, importAttributes: { type: "json" } };
  }
  return nextLoad(url, context);
}
