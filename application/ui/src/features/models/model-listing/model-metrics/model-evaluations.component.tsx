// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid, Text } from '@geti-ui/ui';

import type { Evaluation } from '../../../../constants/shared-types';
import { Box } from '../components/box/box.component';
import { getTestingMetrics } from '../components/model-row/utils';

const formatEvaluationValue = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
};

type ModelEvaluationMetrics = {
    evaluations: Evaluation[];
};

export const ModelEvaluations = ({ evaluations }: ModelEvaluationMetrics) => {
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
                    title={name}
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
