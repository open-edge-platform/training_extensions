// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { buildSSIMInstance } from '@geti/smart-tools';
import { expose, proxy } from 'comlink';

import type { SSIMWorkerApi } from './ssim-worker.interface';

const WorkerApi: SSIMWorkerApi = {
    build: async () => {
        const instance = await buildSSIMInstance();

        return proxy(instance);
    },
};

expose(WorkerApi);
