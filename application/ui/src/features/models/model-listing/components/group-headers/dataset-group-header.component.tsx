// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Flex, Grid, Heading, Text } from '@geti/ui';
import { Image, Tag } from '@geti/ui/icons';

import type { DatasetGroup } from '../../types';
import { ThreeSectionRange } from '../three-section-range.component';

import classes from './group-headers.module.scss';

type DatasetGroupHeaderProps = {
    dataset: DatasetGroup;
};

export const DatasetGroupHeader = ({ dataset }: DatasetGroupHeaderProps) => {
    return (
        <Grid
            columns={['auto', '1fr', 'auto', '1fr', 'auto']}
            alignItems={'center'}
            marginBottom={'size-225'}
            gap={'size-200'}
        >
            <Flex alignItems={'center'} gap={'size-50'}>
                <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                    {dataset.name}
                </Heading>
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

            <ThreeSectionRange
                trainingValue={dataset.trainingSubsets.training}
                validationValue={dataset.trainingSubsets.validation}
                testingValue={dataset.trainingSubsets.testing}
            />

            <Flex>
                <Button variant='primary'>Train model</Button>
            </Flex>
        </Grid>
    );
};
