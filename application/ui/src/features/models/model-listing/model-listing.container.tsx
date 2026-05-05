// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Divider, Flex, Heading } from '@geti/ui';
import { useGetCurrentRunningJob } from 'hooks/api/jobs/jobs.hook';
import { isEmpty, isString } from 'lodash-es';

import { ReactComponent as NoTrainedModels } from '../../../assets/no-trained-models.svg';
import { ExportJobsList } from '../../dataset/import-export/export-jobs-list/export-jobs-list.component';
import { TrainModel } from '../train-model/train-model.component';
import { Header } from './components/header.component';
import { CurrentModelRunning } from './current-model-running/current-model-running.component';
import { ModelListing } from './model-listing.component';
import { ModelListingProvider, useModelListing } from './provider/model-listing-provider';

const ModelListingContent = () => {
    const runningJob = useGetCurrentRunningJob();
    const { groupedModels, searchBy, datasetRevisions, groupBy } = useModelListing();

    const hasNoResults = groupedModels.length === 0 && searchBy.length > 0;
    const hasNoModels = groupedModels.length === 0 && searchBy.length === 0;

    if (hasNoModels) {
        return (
            <Flex
                direction={'column'}
                height={'100%'}
                alignItems={'center'}
                justifyContent={isEmpty(runningJob) ? 'center' : 'start'}
                UNSAFE_style={{ padding: dimensionValue('size-300') }}
            >
                <CurrentModelRunning groupBy={groupBy} datasetRevisions={datasetRevisions} />

                {isEmpty(runningJob) && (
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
                <CurrentModelRunning groupBy={groupBy} datasetRevisions={datasetRevisions} />

                <ExportJobsList predicate={({ datasetId }) => isString(datasetId)} />

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
