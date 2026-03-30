// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';

import type { TrainingConfiguration, TrainingConfigurationParameter } from '../../../../constants/shared-types';
import { isParameterGroup } from '../../model-listing/model-training-parameters/utils';

const createNewPrefix = (prefix: string, key: string) => (isEmpty(prefix) ? key : `${prefix}.${key}`);

const collectConfigurationParameters = (
    parameters: TrainingConfigurationParameter[],
    prefixKey: string
): Record<string, unknown> => {
    return parameters.reduce<Record<string, unknown>>((acc, parameter) => {
        if (isParameterGroup(parameter)) {
            const newPrefix = createNewPrefix(prefixKey, parameter.key);
            const result = collectConfigurationParameters(parameter.parameters, newPrefix);

            return {
                ...acc,
                ...result,
            };
        } else {
            const finalKey = createNewPrefix(prefixKey, parameter.key);
            acc[finalKey] = parameter.value;
        }
        return acc;
    }, {});
};

export const getTrainingConfigurationUpdatePayload = (
    config: TrainingConfiguration | undefined
): Record<string, unknown> => {
    if (!config) {
        return {};
    }

    if (config.parameters.length === 0) {
        return {};
    }

    return collectConfigurationParameters(config.parameters, '');
};
