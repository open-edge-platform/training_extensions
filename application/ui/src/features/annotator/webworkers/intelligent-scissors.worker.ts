// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { buildIntelligentScissorsInstance } from '@geti/smart-tools';
import { expose, proxy } from 'comlink';

const WorkerApi = {
    build: async () => {
        const instance = await buildIntelligentScissorsInstance();

        return proxy(instance);
    },
};

expose(WorkerApi);
