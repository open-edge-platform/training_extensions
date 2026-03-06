// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { IntelligentScissors } from '@geti/smart-tools';
import type { ProxyMarked } from 'comlink';

export type IntelligentScissorsWorkerInstance = IntelligentScissors & ProxyMarked;

export type IntelligentScissorsWorkerApi = {
    build: () => Promise<IntelligentScissorsWorkerInstance>;
};
