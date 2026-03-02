// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';

import type { Evaluation } from '../../../../constants/shared-types';
import { useGetModelTrainingMetrics } from '../../hooks/api/use-get-model-training-metrics.hook';
import { ModelEvaluations } from './model-evaluations.component';
import { ModelMetricsGraphs } from './model-metrics-graphs.component';

type ModelMetricsProps = {
    modelId: string;
    evaluations: Evaluation[];
};

export const ModelMetrics = ({ modelId, evaluations }: ModelMetricsProps) => {
    const { data: trainingMetrics } = useGetModelTrainingMetrics(modelId);

    const testingEvaluations = evaluations.find(({ subset }) => subset === 'testing');
    const testingMetrics = testingEvaluations?.metrics ?? [];

    return (
        <Flex direction='column' gap={'size-300'}>
            <ModelEvaluations metrics={testingMetrics} />

            <ModelMetricsGraphs trainingMetrics={trainingMetrics.training_metrics} />
        </Flex>
    );
};
