// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { TrainingConfiguration } from '../../../configuration.interface';
import { DataAugmentation } from './data-augmentation/data-augmentation.component';
import { Filters } from './filters/filters.component';
import { Tiling } from './tiling/tiling.component';
import { TrainingSubsets } from './training-subsets/training-subsets.component';

type DataManagementProps = {
    defaultTrainingConfiguration: TrainingConfiguration;

    trainingConfiguration: TrainingConfiguration;
    onUpdateTrainingConfiguration: (
        updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
    ) => void;
};

const getAugmentationParameters = (configuration: TrainingConfiguration) => {
    const augmentation = structuredClone(configuration.datasetPreparation.augmentation);

    delete augmentation['tiling'];

    return augmentation;
};

export const DataManagement = ({
    trainingConfiguration,
    onUpdateTrainingConfiguration,
    defaultTrainingConfiguration,
}: DataManagementProps) => {
    const augmentationParameters = getAugmentationParameters(trainingConfiguration);
    const subsetSplitParameters = trainingConfiguration.datasetPreparation.subsetSplit;
    const filteringParameters = trainingConfiguration.datasetPreparation.filtering;
    const tilingParameters = trainingConfiguration.datasetPreparation.augmentation.tiling;

    return (
        <View>
            {!isEmpty(subsetSplitParameters) && (
                <TrainingSubsets
                    defaultSubsetParameters={defaultTrainingConfiguration.datasetPreparation.subsetSplit}
                    subsetsParameters={trainingConfiguration.datasetPreparation.subsetSplit}
                    onUpdateTrainingConfiguration={onUpdateTrainingConfiguration}
                />
            )}
            {!isEmpty(tilingParameters) && (
                <Tiling
                    tilingParameters={tilingParameters}
                    onUpdateTrainingConfiguration={onUpdateTrainingConfiguration}
                />
            )}
            {!isEmpty(augmentationParameters) && (
                <DataAugmentation
                    parameters={augmentationParameters}
                    onUpdateTrainingConfiguration={onUpdateTrainingConfiguration}
                />
            )}
            {!isEmpty(filteringParameters) && (
                <Filters
                    filtersConfiguration={filteringParameters}
                    onUpdateTrainingConfiguration={onUpdateTrainingConfiguration}
                />
            )}
        </View>
    );
};
