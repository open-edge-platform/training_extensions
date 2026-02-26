// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment } from 'react';

import { Divider, Flex, Text } from '@geti/ui';

import { type ExtendedModel } from '../../../../constants/shared-types';
import { useGetModelTrainingMetrics } from '../../hooks/api/use-get-model-training-metrics.hook';
import { ModelMetricsGraphs } from './model-metrics-graphs.component';

type ModelMetricsProps = {
    modelId: string;
    evaluations: ExtendedModel['evaluations'];
};

const formatMetricValue = (value: number): string => {
    if (value >= 0 && value <= 1) {
        return `${(value * 100).toFixed(2)}%`;
    }

    return value.toFixed(4);
};

export const ModelMetrics = ({ modelId, evaluations }: ModelMetricsProps) => {
    const { data: trainingMetrics } = useGetModelTrainingMetrics(modelId);

    const testingEvaluation = evaluations.find(({ subset }) => subset === 'testing');
    const testingMetrics = testingEvaluation?.metrics ?? [];

    return (
        <Flex direction='column' gap={'size-300'}>
            <Flex alignItems={'center'}>
                {testingMetrics.length > 0 ? (
                    testingMetrics.map(({ name, value }, index) => (
                        <Fragment key={name}>
                            {index > 0 && <Divider marginX={'size-300'} orientation={'vertical'} size={'S'} />}
                            <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-900)' }}>
                                {`${name}: ${formatMetricValue(value)}`}
                            </Text>
                        </Fragment>
                    ))
                ) : (
                    <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-900)' }}>
                        Testing evaluation metrics are not available
                    </Text>
                )}
            </Flex>

            <ModelMetricsGraphs trainingMetrics={trainingMetrics.training_metrics} />
        </Flex>
    );
};
