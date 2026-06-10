// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Item, Loading, TabList, TabPanels, Tabs, Text } from '@geti/ui';
import { Info } from '@geti/ui/icons';

import { useGetDatasetRevisions } from '../../../../hooks/use-get-dataset-revisions.hook';
import { UltralyticsLicense } from '../../components/ultralytics-license.component';
import { useGetModel } from '../../hooks/api/use-get-model.hook';
import { isUltralyticsModel } from '../../utils';
import { getModelEvaluations } from '../components/model-row/utils';
import { ModelMetrics } from '../model-metrics/model-metrics.component';
import { ModelTrainingDatasets } from '../model-training-datasets/model-training-datasets.component';
import { ModelTrainingParameters } from '../model-training-parameters/model-training-parameters.component';
import { ModelVariantsTabs } from '../model-variants/model-variant-tabs.component';

interface ModelDetailsTabsProps {
    modelId: string;
}

export const ModelDetailsTabs = ({ modelId }: ModelDetailsTabsProps) => {
    const { isPending, isError, data: model } = useGetModel(modelId);
    const { data: datasetRevisions = [] } = useGetDatasetRevisions();

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

    const currentDatasetRevisionId = model.training_info.dataset_revision_id;

    // Note: currentDatasetRevision might be 'undefined' if the dataset revision was deleted after the model
    // was trained.
    const currentDatasetRevision = datasetRevisions.find(
        (datasetRevision) => datasetRevision.id === currentDatasetRevisionId
    );

    return (
        <Flex direction={'column'} gap={'size-100'}>
            {isUltralyticsModel(model.architecture) && (
                <Flex gap={'size-50'} alignItems={'center'}>
                    <Info />
                    <UltralyticsLicense />
                </Flex>
            )}
            <Tabs
                flex={1}
                minHeight={0}
                aria-label={'Model details'}
                UNSAFE_style={{
                    backgroundColor: 'var(--spectrum-global-color-gray-75)',
                    padding: 'var(--spectrum-global-dimension-size-400)',
                    borderRadius: 'var(--spectrum-global-dimension-size-50)',
                    border: 'var(--spectrum-global-dimension-size-10) solid var(--spectrum-global-color-gray-200)',
                    '--spectrum-tabs-selection-indicator-color': 'var(--energy-blue)',
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
                        <ModelMetrics
                            modelId={model.id}
                            evaluations={getModelEvaluations(model)}
                            filesDeleted={model.files_deleted}
                        />
                    </Item>
                    <Item key='parameters'>
                        <ModelTrainingParameters modelId={model.id} />
                    </Item>
                    <Item key='datasets'>
                        <ModelTrainingDatasets datasetRevision={currentDatasetRevision} />
                    </Item>
                </TabPanels>
            </Tabs>
        </Flex>
    );
};
