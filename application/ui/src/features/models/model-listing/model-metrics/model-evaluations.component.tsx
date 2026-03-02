// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment } from 'react';

import { Divider, Flex, Text } from '@geti/ui';

import type { Metric } from '../../../../constants/shared-types';
import { Box } from '../components/box/box.component';

const formatEvaluationValue = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
};

type ModelEvaluationMetrics = {
    metrics: Metric[];
};

export const ModelEvaluations = ({ metrics }: ModelEvaluationMetrics) => {
    return (
        <Box
            title={'Evaluations'}
            content={
                <Flex alignItems={'center'}>
                    {metrics.length > 0 ? (
                        metrics.map(({ name, value }, index) => (
                            <Fragment key={name}>
                                {index > 0 && <Divider marginX={'size-300'} orientation={'vertical'} size={'S'} />}

                                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-900)' }}>
                                    {`${name}: ${formatEvaluationValue(value)}`}
                                </Text>
                            </Fragment>
                        ))
                    ) : (
                        <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-900)' }}>
                            Testing evaluation metrics are not available
                        </Text>
                    )}
                </Flex>
            }
        />
    );
};
