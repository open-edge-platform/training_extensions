// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Default browser-build timeouts for the Segment Anything ONNX worker.
// WebGPU is available in modern browsers and brings encoder runs down to ~1s
// (see /memories/repo/onnxruntime-web-perf-baseline.md), so these values fail
// fast on real hangs without giving up on legitimately slow first runs.
//
// Tauri overrides these via `sam-timeouts.tauri.ts` (resolved by the
// `.tauri.*` extension list in rsbuild.config.ts) because WKWebView/WebView2
// either do not support WebGPU or fall back to CPU, where the same encoder
// can take 5–25s on slower machines.

export const SAM_DECODER_TIMEOUT_MS = 20_000;
export const SAM_ENCODER_TIMEOUT_MS = 30_000;
export const SAM_WORKER_BUILD_TIMEOUT_MS = 10_000;
export const SAM_WORKER_INIT_TIMEOUT_MS = SAM_ENCODER_TIMEOUT_MS;

// How long an unobserved encoding (a large Float32 tensor per image) is kept
// in the query cache before being garbage-collected. Short on purpose:
// encodings are heavy and are cheap-ish to recompute, so we favor memory over
// perf.
export const SAM_ENCODING_GC_TIME_MS = 60_000;
