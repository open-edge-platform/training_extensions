// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { buildSSIMInstance } from '@geti/smart-tools';
import { expose, proxy } from 'comlink';

const WorkerApi = {
    build: async () => {
        const instance = await buildSSIMInstance();

        return proxy(instance);
    },
};

expose(WorkerApi);
