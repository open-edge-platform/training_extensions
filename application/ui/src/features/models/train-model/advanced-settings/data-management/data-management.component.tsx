// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { View } from '@geti/ui';

import { TrainingConfiguration } from '../../../../../constants/shared-types';
import { LazyLoadSection } from '../components/lazy-load-section.component';
import { DataAugmentation } from './data-augmentation/data-augmentation.component';
import { getDataAugmentationParameters } from './data-augmentation/utils';
import { Filters } from './filters/filters.component';
import { getFiltersParameters } from './filters/utils';
import { Tiling } from './tiling/tiling.component';
import { getTilingParameters } from './tiling/utils';
import { TrainingSubsets } from './training-subsets/training-subsets.component';
import { getSubsetSplitParameters } from './training-subsets/utils';

type DataManagementProps = {
    trainingConfiguration: TrainingConfiguration;
    defaultTrainingConfiguration: TrainingConfiguration;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

export const DataManagement = ({
    trainingConfiguration,
    defaultTrainingConfiguration,
    onTrainingConfigurationChange,
}: DataManagementProps) => {
    const subsetSplitParameters = getSubsetSplitParameters(trainingConfiguration);
    const defaultSubsetSplitParameters = getSubsetSplitParameters(defaultTrainingConfiguration);

    const filtersParameters = getFiltersParameters(trainingConfiguration);
    const tilingParameters = getTilingParameters(trainingConfiguration);
    const dataAugmentationParameters = getDataAugmentationParameters(trainingConfiguration);

    return (
        <View>
            {subsetSplitParameters !== undefined && defaultSubsetSplitParameters !== undefined && (
                <TrainingSubsets
                    defaultSubsetParameters={defaultSubsetSplitParameters}
                    subsetsParameters={subsetSplitParameters}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            )}

            {tilingParameters !== undefined && (
                <Tiling
                    tilingParameters={tilingParameters}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            )}

            {dataAugmentationParameters !== undefined && (
                <LazyLoadSection>
                    <DataAugmentation
                        dataAugmentationParameters={dataAugmentationParameters}
                        onTrainingConfigurationChange={onTrainingConfigurationChange}
                    />
                </LazyLoadSection>
            )}

            {filtersParameters !== undefined && (
                <LazyLoadSection>
                    <Filters
                        filtersParameters={filtersParameters}
                        onTrainingConfigurationChange={onTrainingConfigurationChange}
                    />
                </LazyLoadSection>
            )}
        </View>
    );
};
