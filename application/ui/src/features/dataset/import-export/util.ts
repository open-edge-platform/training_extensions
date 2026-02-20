// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const getJobProgress = (progress?: number) => Math.floor(Math.max(0, Math.min(100, progress ?? 0)));
