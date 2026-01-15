// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, TabList, TabPanels, Tabs, Text } from '@geti/ui';

import type { SchemaModelView } from '../../../../api/openapi-spec';
import { ModelMetrics } from '../model-metrics/model-metrics.component';
import { ModelTrainingDatasets } from '../model-training-datasets/model-training-datasets.component';
import { ModelTrainingParameters } from '../model-training-parameters/model-training-parameters.component';
import { ModelVariantsTabs } from '../model-variants/model-variant-tabs.component';

interface ModelDetailsTabsProps {
    model: SchemaModelView;
}

export const ModelDetailsTabs = ({ model }: ModelDetailsTabsProps) => {
    return (
        <Tabs
            aria-label={'Model details'}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-75)',
                padding: 'var(--spectrum-global-dimension-size-400)',
                borderRadius: 'var(--spectrum-global-dimension-size-50)',
                border: 'var(--spectrum-global-dimension-size-10) solid var(--spectrum-global-color-gray-200)',
            }}
        >
            <TabList marginBottom={'size-300'}>
                <Item key='variants'>
                    <Text>Model variants</Text>
                </Item>
                <Item key='metrics'>
                    <Text>Model metrics</Text>
                </Item>
                <Item key='parameters'>
                    <Text>Training parameters</Text>
                </Item>
                <Item key='datasets'>
                    <Text>Training datasets</Text>
                </Item>
            </TabList>
            <TabPanels>
                <Item key='variants'>
                    <ModelVariantsTabs model={model} />
                </Item>
                <Item key='metrics'>
                    <ModelMetrics />
                </Item>
                <Item key='parameters'>
                    <ModelTrainingParameters />
                </Item>
                <Item key='datasets'>
                    <ModelTrainingDatasets />
                </Item>
            </TabPanels>
        </Tabs>
    );
};
