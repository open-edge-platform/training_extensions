// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePipeline } from 'hooks/api/pipeline.hook';

export const useGetActiveModel = () => {
    const pipeline = usePipeline();

    return {
        ...pipeline.data.model,
        model_variant_id: pipeline.data.model_variant?.id,
    };
};
