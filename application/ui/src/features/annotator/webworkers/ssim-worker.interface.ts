// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SSIM } from '@geti/smart-tools';
import type { ProxyMarked } from 'comlink';

export type SSIMWorkerInstance = SSIM & ProxyMarked;

export type SSIMWorkerApi = {
    build: () => Promise<SSIMWorkerInstance>;
};
