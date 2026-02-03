// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Grid, Heading, Text } from '@geti/ui';
import { Image, Tag } from '@geti/ui/icons';

import { TrainModel } from '../../../train-model/train-model.component';
import type { DatasetGroup } from '../../types';
import { DatasetActions } from '../dataset-actions/dataset-actions.component';
import { ThreeSectionRange } from '../three-section-range/three-section-range.component';

import classes from './group-headers.module.scss';

type DatasetGroupHeaderProps = {
    dataset: DatasetGroup;
};

export const DatasetGroupHeader = ({ dataset }: DatasetGroupHeaderProps) => {
    const gridColumns = dataset.filesDeleted ? ['auto', '1fr', 'auto', 'auto'] : ['auto', '1fr', 'auto', '1fr', 'auto'];

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
                <Flex UNSAFE_className={classes.tag}>
                    <Tag /> {dataset.labelCount}
                </Flex>
                <Flex UNSAFE_className={classes.tag}>
                    <Image /> {dataset.imageCount.toLocaleString()}
                </Flex>
            </Flex>

            {!dataset.filesDeleted && (
                <ThreeSectionRange
                    id={`dataset-range-${dataset.id}`}
                    trainingValue={dataset.trainingSubsets.training}
                    validationValue={dataset.trainingSubsets.validation}
                    testingValue={dataset.trainingSubsets.testing}
                />
            )}

            <Flex>
                <TrainModel preSelectedDatasetRevisionId={dataset.id} />
            </Flex>
        </Grid>
    );
};
