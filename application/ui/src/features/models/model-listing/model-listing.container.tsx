// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Divider, Flex, Heading } from '@geti/ui';
import { useGetCurrentTrainingJob } from 'hooks/api/jobs.hook';
import { isEmpty } from 'lodash-es';

import { ReactComponent as NoTrainedModels } from '../../../assets/no-trained-models.svg';
import { TrainModel } from '../train-model/train-model.component';
import { Header } from './components/header.component';
import { CurrentModelTraining } from './current-model-training/current-model-training.component';
import { ModelListing } from './model-listing.component';
import { ModelListingProvider, useModelListing } from './provider/model-listing-provider';

const ModelListingContent = () => {
    const { groupedModels, searchBy } = useModelListing();
    const trainingJob = useGetCurrentTrainingJob();

    const hasNoResults = groupedModels.length === 0 && searchBy.length > 0;
    const hasNoModels = groupedModels.length === 0 && searchBy.length === 0;

    if (hasNoModels) {
        return (
            <Flex
                direction={'column'}
                height={'100%'}
                alignItems={'center'}
                UNSAFE_style={{ padding: dimensionValue('size-300') }}
            >
                <CurrentModelTraining />

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

            <CurrentModelTraining />

            <ModelListing hasNoResults={hasNoResults} groupedModels={groupedModels} />
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
