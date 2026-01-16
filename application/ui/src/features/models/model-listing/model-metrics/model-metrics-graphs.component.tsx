// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';

import { MetricGraph } from './metric-graph.component';

type ModelMetricsGraphsProps = {
    // TODO: update types
    lossData?: { epoch: number; trainLoss: number }[];
    accuracyData?: { epoch: number; trainAccuracy: number }[];
};

export const ModelMetricsGraphs = ({ lossData, accuracyData }: ModelMetricsGraphsProps) => {
    return (
        <Flex width={'100%'} direction={'row'} gap={'size-300'}>
            <MetricGraph
                title='ACCURACY'
                data={accuracyData}
                dataKey='trainAccuracy'
                yAxisLabel='train accuracy'
                yAxisDomain={[0, 1]}
                yAxisTicks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
            />
            <MetricGraph title='LOSS' data={lossData} dataKey='trainLoss' yAxisLabel='train loss' />
        </Flex>
    );
};
