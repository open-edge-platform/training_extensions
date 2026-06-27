// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ConfigurableParameterGroup,
    TrainingConfiguration,
} from '../../../../../../constants/shared-types';
import { findGroupByKey } from '../../../../model-listing/model-training-parameters/utils';

export type EvaluationConfigurationGroup = ConfigurableParameterGroup;

export const getEvaluationParameters = (
    trainingConfiguration: TrainingConfiguration
): EvaluationConfigurationGroup | undefined => {
    return findGroupByKey(trainingConfiguration.parameters, 'evaluation');
};
