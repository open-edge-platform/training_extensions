// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Badge, Flex, Grid, Text, Tooltip, TooltipTrigger } from '@geti/ui';
import { InfoOutline } from '@geti/ui/icons';

import type { Evaluation, TrainingConfiguration } from '../../../../constants/shared-types';
import { Box } from '../components/box/box.component';
import { getTestingMetrics } from '../components/model-row/utils';
import { findGroupByKey } from '../model-training-parameters/utils';

const formatEvaluationValue = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
};

type ModelEvaluationMetrics = {
    evaluations: Evaluation[];
    trainingConfiguration?: TrainingConfiguration;
};

const metricDescriptions: Record<string, string> = {
    Accuracy: 'Accuracy measures the overall correctness of the model across all classes.',
    Precision:
        'Precision indicates how many of the positively predicted instances were actually correct. ' +
        'High precision minimizes false positives.',
    Recall:
        'Recall indicates how many of the actual positive instances were successfully predicted. ' +
        'High recall minimizes false negatives.',
    'F-measure': 'F-measure (or F1 Score) is the harmonic mean of precision and recall, balancing the two.',
    mAP:
        'mAP (Mean Average Precision) evaluates detection and segmentation models ' +
        'by averaging precision across recall levels.',
    'mAP@0.5': 'mAP at an Intersection over Union (IoU) threshold of 0.5.',
    'mAP@0.75': 'mAP at an Intersection over Union (IoU) threshold of 0.75.',
};

export const ModelEvaluations = ({ evaluations, trainingConfiguration }: ModelEvaluationMetrics) => {
    const evaluationGroup = trainingConfiguration
        ? findGroupByKey(trainingConfiguration.parameters, 'evaluation')
        : undefined;
    const validationMetricParam = evaluationGroup?.parameters.find(
        (p) => p.type === 'parameter' && p.key === 'validation_metric'
    );
    const validationMetric = validationMetricParam?.type === 'parameter' ? validationMetricParam.value : undefined;
    const testingMetrics = getTestingMetrics(evaluations);

    if (testingMetrics.length === 0) {
        return (
            <Box
                title={'Evaluations'}
                content={
                    <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-900)' }}>
                        Testing evaluation metrics are not available
                    </Text>
                }
            />
        );
    }

    return (
        <Grid columns={['1fr', '1fr', '1fr']} gap={'size-200'}>
            {testingMetrics.map(({ name, value }) => (
                <Box
                    key={name}
                    title={
                        <Flex alignItems='center' gap='size-50'>
                            {name}
                            <TooltipTrigger delay={0}>
                                <InfoOutline
                                    UNSAFE_style={{
                                        color: 'var(--spectrum-global-color-gray-500)',
                                        cursor: 'default',
                                        height: '16px',
                                        width: '16px',
                                    }}
                                />
                                <Tooltip>{metricDescriptions[name] || 'Performance metric'}</Tooltip>
                            </TooltipTrigger>
                            {validationMetric !== undefined &&
                                (validationMetric === name ||
                                    (validationMetric === 'default' &&
                                        (name === 'Accuracy' ||
                                            name === 'F-measure' ||
                                            name === 'mAP' ||
                                            name === 'mAP@0.5'))) && (
                                    <Badge variant='info' UNSAFE_style={{ marginLeft: 'auto' }}>
                                        Optimized for
                                    </Badge>
                                )}
                        </Flex>
                    }
                    content={
                        <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-900)' }}>
                            {formatEvaluationValue(value)}
                        </Text>
                    }
                />
            ))}
        </Grid>
    );
};
