// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';

import type { components } from '../api/openapi-spec';

type Pipeline = components['schemas']['PipelineView'];
export const useIsPipelineConfigured = (pipeline: Pipeline) => {
    if (!pipeline) return false;

    const { model, source, sink } = pipeline;
    const isEditable = !isEmpty(model) && !isEmpty(source) && !isEmpty(sink);

    return isEditable;
};
