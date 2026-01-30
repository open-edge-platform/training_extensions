// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Text } from '@geti/ui';

import { ModelMetricsGraphs } from './model-metrics-graphs.component';

// TODO: replace this with actual API data
const mockAccuracyData = [
    { epoch: 0, trainAccuracy: 0.1 },
    { epoch: 3, trainAccuracy: 0.35 },
    { epoch: 6, trainAccuracy: 0.52 },
    { epoch: 9, trainAccuracy: 0.68 },
    { epoch: 12, trainAccuracy: 0.78 },
    { epoch: 15, trainAccuracy: 0.85 },
    { epoch: 18, trainAccuracy: 0.89 },
    { epoch: 21, trainAccuracy: 0.92 },
    { epoch: 24, trainAccuracy: 0.94 },
    { epoch: 27, trainAccuracy: 0.95 },
    { epoch: 30, trainAccuracy: 0.96 },
    { epoch: 33, trainAccuracy: 0.965 },
    { epoch: 36, trainAccuracy: 0.97 },
    { epoch: 39, trainAccuracy: 0.975 },
    { epoch: 43, trainAccuracy: 0.98 },
];

const mockLossData = [
    { epoch: 0, trainLoss: 2.5 },
    { epoch: 3, trainLoss: 1.8 },
    { epoch: 6, trainLoss: 1.2 },
    { epoch: 9, trainLoss: 0.8 },
    { epoch: 12, trainLoss: 0.5 },
    { epoch: 15, trainLoss: 0.35 },
    { epoch: 18, trainLoss: 0.25 },
    { epoch: 21, trainLoss: 0.18 },
    { epoch: 24, trainLoss: 0.12 },
    { epoch: 27, trainLoss: 0.09 },
    { epoch: 30, trainLoss: 0.07 },
    { epoch: 33, trainLoss: 0.055 },
    { epoch: 36, trainLoss: 0.045 },
    { epoch: 39, trainLoss: 0.038 },
    { epoch: 43, trainLoss: 0.03 },
];

export const ModelMetrics = () => {
    const trainingTime = '00:02:47';
    const jobDuration = '00:04:35';

    return (
        <Flex direction='column' gap={'size-300'}>
            <Flex alignItems={'center'}>
                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-900)' }}>
                    {`Model training time: ${trainingTime}`}
                </Text>
                <Divider marginX={'size-300'} orientation={'vertical'} size={'S'} />
                <Text
                    UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-900)' }}
                >{`Job duration: ${jobDuration}`}</Text>
            </Flex>

            <ModelMetricsGraphs lossData={mockLossData} accuracyData={mockAccuracyData} />
        </Flex>
    );
};
