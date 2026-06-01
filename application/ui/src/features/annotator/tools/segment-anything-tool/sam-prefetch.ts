// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Controls whether the SAM encoder is speculatively prefetched for the next
// media item while the current frame is being annotated. Browsers (WebGPU
// path, ~1 s encoder runs) tolerate the extra parallel job comfortably, so
// the prefetch is a worthwhile latency win when the user advances to the
// next frame. Tauri overrides this via `sam-prefetch.tauri.ts` (resolved by
// the `.tauri.*` extension list in rsbuild.config.ts) — on the CPU EP the
// prefetch competes with the current encode on the same single worker and
// rarely lands before the user moves on, net-negative on throughput AND
// the dominant contributor to the worker-mailbox backlog that produces the
// 30 s "SAM encoder timed out" symptom.

export const SAM_NEXT_FRAME_PREFETCH = true;
