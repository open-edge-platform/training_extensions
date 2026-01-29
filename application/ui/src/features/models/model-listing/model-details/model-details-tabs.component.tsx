// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Item, Loading, TabList, TabPanels, Tabs, Text } from '@geti/ui';

import { useGetModel } from '../../hooks/api/use-get-model.hook';
import { ModelMetrics } from '../model-metrics/model-metrics.component';
import { ModelTrainingDatasets } from '../model-training-datasets/model-training-datasets.component';
import { ModelTrainingParameters } from '../model-training-parameters/model-training-parameters.component';
import { ModelVariantsTabs } from '../model-variants/model-variant-tabs.component';

interface ModelDetailsTabsProps {
    modelId: string;
}

export const ModelDetailsTabs = ({ modelId }: ModelDetailsTabsProps) => {
    const { isPending, isError, data: model } = useGetModel(modelId);

    if (isPending) {
        return (
            <Flex alignItems={'center'} justifyContent={'center'} height={'size-3000'}>
                <Loading size={'M'} />
            </Flex>
        );
    }

    if (isError || !model) {
        return (
            <Flex alignItems={'center'} justifyContent={'center'} height={'size-3000'}>
                <Text>Failed to load model details</Text>
            </Flex>
        );
    }

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
                    <ModelTrainingDatasets datasetRevisionId={model.training_info.dataset_revision_id} />
                </Item>
            </TabPanels>
        </Tabs>
    );
};
