// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti/ui';

type SubsetDistributionRowProps = {
    label: string;
    existingSize: number;
    newSize: number;
    totalSize: number;
};

const SubsetDistributionRow = ({ label, existingSize, newSize, totalSize }: SubsetDistributionRowProps) => {
    const resultingSize = existingSize + newSize;
    const percentage = totalSize > 0 ? Math.round((resultingSize * 100) / totalSize) : 0;

    return (
        <Text>
            {label}: {existingSize} + {newSize} = {resultingSize} ({percentage}%)
        </Text>
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
        <Flex direction={'column'}>
            <Text>Resulting dataset distribution:</Text>
            <SubsetDistributionRow
                label={'Training'}
                existingSize={trainingSubsetSize}
                newSize={newTrainingSubsetSize}
                totalSize={totalDatasetItemsSize}
            />
            <SubsetDistributionRow
                label={'Validation'}
                existingSize={validationSubsetSize}
                newSize={newValidationSubsetSize}
                totalSize={totalDatasetItemsSize}
            />
            <SubsetDistributionRow
                label={'Testing'}
                existingSize={testingSubsetSize}
                newSize={newTestingSubsetSize}
                totalSize={totalDatasetItemsSize}
            />
        </Flex>
    );
};
