// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const x = Math.floor(cb_obj.x), y = Math.floor(cb_obj.y);
if (x < 0 || y < 0 || x >= size || y >= size) return;
exact.text = `Raw electrons at (${x}, ${y}): ` + sources.map(
    s => String(s.data.raw_flat[0][y * size + x])
).join(' | ');
