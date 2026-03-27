// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Grid, Heading, Text } from '@geti/ui';
import { Image, Tag } from '@geti/ui/icons';
import { useNumberFormatter } from 'react-aria';

import type { DatasetGroup } from '../../types';
import { DatasetActions } from '../dataset-actions/dataset-actions.component';
import { ModelBadge } from '../model-row/model-badge.component';
import { ThreeSectionRange } from '../three-section-range/three-section-range.component';

type DatasetGroupHeaderProps = {
    dataset: DatasetGroup;
};

export const DatasetGroupHeader = ({ dataset }: DatasetGroupHeaderProps) => {
    const hasDatasetRevisionData = dataset.imageCount > 0 && !dataset.filesDeleted;
    const gridColumns = hasDatasetRevisionData ? ['auto', '1fr', 'auto', '1fr'] : ['auto', '1fr', 'auto'];
    const formatter = useNumberFormatter();

    const subsetsTotal =
        dataset.trainingSubsets.training + dataset.trainingSubsets.validation + dataset.trainingSubsets.testing;

    const percentages = {
        training: subsetsTotal > 0 ? (dataset.trainingSubsets.training / subsetsTotal) * 100 : 0,
        validation: subsetsTotal > 0 ? (dataset.trainingSubsets.validation / subsetsTotal) * 100 : 0,
        testing: subsetsTotal > 0 ? (dataset.trainingSubsets.testing / subsetsTotal) * 100 : 0,
    };

    return (
        <Grid columns={gridColumns} alignItems={'center'} marginBottom={'size-225'} gap={'size-200'}>
            <Flex alignItems={'center'} gap={'size-50'}>
                <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                    {dataset.name}
                </Heading>

                <DatasetActions dataset={dataset} />
            </Flex>
            <Text
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-700)',
                }}
            >
                {dataset.createdAt}
            </Text>

            <Flex gap={'size-50'} justifyContent={'center'}>
                <ModelBadge>
                    <Tag /> {dataset.labelCount}
                </ModelBadge>
                <ModelBadge>
                    <Image /> {formatter.format(dataset.imageCount)}
                </ModelBadge>
            </Flex>

            {hasDatasetRevisionData && (
                <ThreeSectionRange
                    id={`dataset-range-${dataset.id}`}
                    trainingValue={percentages.training}
                    validationValue={percentages.validation}
                    testingValue={percentages.testing}
                />
            )}
        </Grid>
    );
};
