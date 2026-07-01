// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Loading, Text } from '@geti/ui';

import type { Evaluation } from '../../../../constants/shared-types';
import { useGetModelTrainingMetrics } from '../../hooks/api/use-get-model-training-metrics.hook';
import { ModelEvaluations } from './model-evaluations.component';
import { ModelMetricsGraphs } from './model-metrics-graphs.component';

type ModelMetricsProps = {
    modelId: string;
    evaluations: Evaluation[];
    filesDeleted?: boolean;
};

export const ModelMetrics = ({ modelId, evaluations, filesDeleted = false }: ModelMetricsProps) => {
    const { data: trainingMetrics, isPending, isError } = useGetModelTrainingMetrics(filesDeleted ? null : modelId);

    if (filesDeleted) {
        return (
            <Flex alignItems={'center'} justifyContent={'center'} height={'size-3000'}>
                <Text>No available metrics</Text>
            </Flex>
        );
    }

    return (
        <Flex direction='column' gap={'size-300'}>
            {isPending ? (
                <Flex alignItems={'center'} justifyContent={'center'} height={'size-3000'}>
                    <Loading size={'M'} mode={'inline'} />
                </Flex>
            ) : isError ? (
                <Flex alignItems={'center'} justifyContent={'center'} height={'size-3000'}>
                    <Text>Failed to load training metrics</Text>
                </Flex>
            ) : (
                <>
                    <ModelEvaluations evaluations={evaluations} />
                    <ModelMetricsGraphs trainingMetrics={trainingMetrics?.training_metrics ?? []} />
                </>
            )}
        </Flex>
    );
};
