// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { buildSegmentAnythingInstance } from '@geti/smart-tools/segment-anything';
import type { ProxyMarked } from 'comlink';

export type SegmentAnythingWorkerModel = Awaited<ReturnType<typeof buildSegmentAnythingInstance>> & ProxyMarked;

export type SegmentAnythingWorkerApi = {
    build: () => Promise<SegmentAnythingWorkerModel>;
};
