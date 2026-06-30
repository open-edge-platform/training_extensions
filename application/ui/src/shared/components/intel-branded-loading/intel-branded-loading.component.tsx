// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Browser builds re-export the canonical, animated-WebP version from @geti-ui/ui.
// Tauri builds shadow this file via `intel-branded-loading.component.tauri.tsx`
// (resolved through the `.tauri.*` extension list in rsbuild.config.ts) with a
// pure-CSS spinner so the system webview doesn't pay the per-frame WebP decode
// cost — that decoding stalls the same main thread that's loading WASM modules
// and OpenCV during SAM init, producing the visible jank.

export { IntelBrandedLoading } from '@geti-ui/ui';
