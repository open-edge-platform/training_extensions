// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, repeat, Text, View } from '@geti/ui';

import { LABEL_COLOR_MAPPING, SubsetTile } from './subset-distribution-stats.component';

import classes from './training-subsets.module.scss';

type SubsetDistributionRowProps = {
    existingSize: number;
    newSize: number;
    totalSize: number;
    label: string;
};

const SubsetLabel = ({ label, color }: { color: string; label: string }) => {
    return (
        <Flex alignItems={'center'} gap={'size-50'}>
            <SubsetTile color={color} />
            <Text>{label}:</Text>
        </Flex>
    );
};

const SubsetDistributionRow = ({ existingSize, newSize, totalSize, label }: SubsetDistributionRowProps) => {
    const resultingSize = existingSize + newSize;
    const percentage = totalSize > 0 ? Math.round((resultingSize * 100) / totalSize) : 0;

    return (
        <>
            <Text>{existingSize}</Text>
            <Text>+</Text>
            <Text>{newSize}</Text>
            <Text>=</Text>
            <span aria-label={`${label} result size`}>{resultingSize}</span>
            <span aria-label={`${label} result percentage`}>({percentage}%)</span>
        </>
    );
};

type ResultingDatasetDistributionSubsetProps = {
    color: string;
    label: string;
    totalSize: number;
    newSize: number;
    existingSize: number;
};

const ResultingDatasetDistributionSubset = ({
    color,
    label,
    existingSize,
    newSize,
    totalSize,
}: ResultingDatasetDistributionSubsetProps) => {
    return (
        <>
            <SubsetLabel label={label} color={color} />

            <SubsetDistributionRow label={label} totalSize={totalSize} newSize={newSize} existingSize={existingSize} />
        </>
    );
};

type ResultingDatasetDistributionProps = {
    trainingSubsetSize: number;
    validationSubsetSize: number;
    testingSubsetSize: number;
    newTrainingSubsetSize: number;
    newValidationSubsetSize: number;
    newTestingSubsetSize: number;
    totalDatasetItemsSize: number;
};

export const ResultingDatasetDistribution = ({
    trainingSubsetSize,
    validationSubsetSize,
    testingSubsetSize,
    newTrainingSubsetSize,
    newValidationSubsetSize,
    newTestingSubsetSize,
    totalDatasetItemsSize,
}: ResultingDatasetDistributionProps) => {
    return (
        <Flex direction={'column'} gap={'size-50'}>
            <Text>Resulting dataset distribution:</Text>
            <View backgroundColor={'static-gray-800'} borderRadius={'small'} padding={'size-100'}>
                <Grid
                    columns={[repeat(7, 'max-content')]}
                    alignItems={'center'}
                    columnGap={'size-100'}
                    rowGap={'size-50'}
                    UNSAFE_className={classes.resultingDistributionText}
                >
                    <ResultingDatasetDistributionSubset
                        label={'Training'}
                        color={LABEL_COLOR_MAPPING.training}
                        totalSize={totalDatasetItemsSize}
                        newSize={newTrainingSubsetSize}
                        existingSize={trainingSubsetSize}
                    />

                    <ResultingDatasetDistributionSubset
                        label={'Validation'}
                        color={LABEL_COLOR_MAPPING.validation}
                        totalSize={totalDatasetItemsSize}
                        newSize={newValidationSubsetSize}
                        existingSize={validationSubsetSize}
                    />

                    <ResultingDatasetDistributionSubset
                        label={'Test'}
                        color={LABEL_COLOR_MAPPING.test}
                        totalSize={totalDatasetItemsSize}
                        newSize={newTestingSubsetSize}
                        existingSize={testingSubsetSize}
                    />
                </Grid>
            </View>
        </Flex>
    );
};
