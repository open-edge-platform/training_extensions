// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ConfigurableParameterGroup, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { findGroupByKey } from '../../../../model-listing/model-training-parameters/utils';

export const getIntensityMappingParameters = (
    trainingConfiguration: TrainingConfiguration
): ConfigurableParameterGroup | undefined => {
    const datasetPreparation = findGroupByKey(trainingConfiguration.parameters, 'dataset_preparation')?.parameters;
    return findGroupByKey(datasetPreparation, 'intensity_mapping');
};
