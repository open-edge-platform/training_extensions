// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Tauri override: disable the SAM next-frame encoder prefetch. WKWebView /
// WebView2 fall back to the CPU EP for onnxruntime-web, where SAM-Mobile
// encoding at 1024² runs 5–25 s. Prefetching means a second 5–25 s job is
// queued behind the current one on the single SAM worker — on toggles
// between Annotation and Prediction the stale jobs accumulate and push the
// newest call's wait past `SAM_ENCODER_TIMEOUT_MS`. Disabling the prefetch
// keeps the worker mailbox shallow.

export const SAM_NEXT_FRAME_PREFETCH = false;
