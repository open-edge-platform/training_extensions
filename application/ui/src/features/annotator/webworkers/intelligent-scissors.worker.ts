// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { buildIntelligentScissorsInstance } from '@geti/smart-tools';
import { expose, proxy } from 'comlink';

import type { IntelligentScissorsWorkerApi } from './intelligent-scissors.worker.interface';

const WorkerApi: IntelligentScissorsWorkerApi = {
    build: async () => {
        const instance = await buildIntelligentScissorsInstance();

        return proxy(instance);
    },
};

expose(WorkerApi);
