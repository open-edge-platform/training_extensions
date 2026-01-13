// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePipeline } from 'hooks/api/pipeline.hook';

export const useGetActiveModelId = () => {
    const pipeline = usePipeline();

    return pipeline.data.model?.id;
};
