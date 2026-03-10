// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SegmentAnythingModelWrapper } from '@geti/smart-tools/segment-anything';
import type { ProxyMarked } from 'comlink';

export type SegmentAnythingWorkerInstance = SegmentAnythingModelWrapper & ProxyMarked;

export type SegmentAnythingWorkerApi = {
    build: () => Promise<SegmentAnythingWorkerInstance>;
};
