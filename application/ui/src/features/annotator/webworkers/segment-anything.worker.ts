// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { buildSegmentAnythingInstance } from '@geti/smart-tools/segment-anything';
import { expose, proxy } from 'comlink';

const WorkerApi = {
    build: async () => {
        const instance = await buildSegmentAnythingInstance();

        return proxy(instance);
    },
};

expose(WorkerApi);
