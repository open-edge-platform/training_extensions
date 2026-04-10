// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ConfigurableParameterGroup,
    NumberEnumConfigurableParameter,
    TrainingConfiguration,
    TrainingConfigurationParameter,
} from '../../../../../../constants/shared-types';
import { findGroupByKey, isParameter } from '../../../../model-listing/model-training-parameters/utils';
import { isEnumNumberParameter } from '../../utils';

export type LearningConfigurationGroup = ConfigurableParameterGroup;

export const getLearningParameters = (
    trainingConfiguration: TrainingConfiguration
): LearningConfigurationGroup | undefined => {
    return findGroupByKey(trainingConfiguration.parameters, 'training');
};

export const isInputSizeWidthParameter = (
    parameter: TrainingConfigurationParameter
): parameter is NumberEnumConfigurableParameter =>
    isParameter(parameter) && isEnumNumberParameter(parameter) && parameter.key === 'input_size_width';

export const isInputSizeHeightParameter = (
    parameter: TrainingConfigurationParameter
): parameter is NumberEnumConfigurableParameter =>
    isParameter(parameter) && isEnumNumberParameter(parameter) && parameter.key === 'input_size_height';

export const isInputSizeParameter = (parameter: TrainingConfigurationParameter) =>
    isInputSizeWidthParameter(parameter) || isInputSizeHeightParameter(parameter);

export const getInputSizeWidthParameter = (
    parameters: TrainingConfigurationParameter[]
): NumberEnumConfigurableParameter | undefined => parameters.find(isInputSizeWidthParameter);

export const getInputSizeHeightParameter = (
    parameters: TrainingConfigurationParameter[]
): NumberEnumConfigurableParameter | undefined => parameters.find(isInputSizeHeightParameter);
