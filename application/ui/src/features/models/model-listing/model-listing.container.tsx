// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Divider, Flex, Heading } from '@geti/ui';
import { useGetCurrentTrainingJob } from 'hooks/api/jobs.hook';
import { isEmpty, isString } from 'lodash-es';

import { ReactComponent as NoTrainedModels } from '../../../assets/no-trained-models.svg';
import { ExportJobsList } from '../../dataset/import-export/export-jobs-list/export-jobs-list.component';
import { TrainModel } from '../train-model/train-model.component';
import { Header } from './components/header.component';
import { CurrentModelTraining } from './current-model-training/current-model-training.component';
import { ModelListing } from './model-listing.component';
import { ModelListingProvider, useModelListing } from './provider/model-listing-provider';

const ModelListingContent = () => {
    const { groupedModels, searchBy, datasetRevisions, groupBy } = useModelListing();
    const trainingJob = useGetCurrentTrainingJob();

    const hasNoResults = groupedModels.length === 0 && searchBy.length > 0;
    const hasNoModels = groupedModels.length === 0 && searchBy.length === 0;

    if (hasNoModels) {
        return (
            <Flex
                direction={'column'}
                height={'100%'}
                alignItems={'center'}
                justifyContent={isEmpty(trainingJob) ? 'center' : 'start'}
                UNSAFE_style={{ padding: dimensionValue('size-300') }}
            >
                <CurrentModelTraining groupBy={groupBy} datasetRevisions={datasetRevisions} />

                {isEmpty(trainingJob) && (
                    <Flex direction={'column'} alignItems={'center'} gap={'size-100'} marginTop={'size-600'}>
                        <NoTrainedModels />
                        <Heading level={2}>No models yet. Train your first model to get started.</Heading>
                        <TrainModel />
                    </Flex>
                )}
            </Flex>
        );
    }

    return (
        <Flex direction={'column'} height={'100%'} UNSAFE_style={{ padding: dimensionValue('size-300') }}>
            <Header />

            <Divider size={'S'} marginY={'size-300'} />

            <Flex direction={'column'} flex={1} UNSAFE_style={{ overflowY: 'auto', scrollbarGutter: 'stable' }}>
                <CurrentModelTraining groupBy={groupBy} datasetRevisions={datasetRevisions} />

                <ExportJobsList predicate={({ datasetId }: { datasetId: string | null }) => isString(datasetId)} />

                <ModelListing hasNoResults={hasNoResults} groupedModels={groupedModels} />
            </Flex>
        </Flex>
    );
};

export const ModelListingContainer = () => {
    return (
        <ModelListingProvider>
            <ModelListingContent />
        </ModelListingProvider>
    );
};
