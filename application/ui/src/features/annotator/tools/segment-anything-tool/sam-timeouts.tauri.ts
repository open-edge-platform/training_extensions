// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Tauri-specific SAM timeout budgets, picked up automatically via the
// `.tauri.*` resolver in rsbuild.config.ts. Tauri ships a system webview
// (WKWebView on macOS, WebView2 on Windows, WebKitGTK on Linux); WebGPU is
// either unavailable or behind flags, so onnxruntime-web falls back to CPU.
// CPU SAM-Mobile encoding at 1024² is routinely 5–25s on average laptops,
// which makes the 30s browser timeout produce false positives even on a
// healthy worker. Larger budgets keep slow runs alive.

export const SAM_DECODER_TIMEOUT_MS = 30_000;
export const SAM_ENCODER_TIMEOUT_MS = 60_000;
export const SAM_WORKER_BUILD_TIMEOUT_MS = 30_000;
export const SAM_WORKER_INIT_TIMEOUT_MS = SAM_ENCODER_TIMEOUT_MS;

export const SAM_ENCODING_GC_TIME_MS = 60_000;
