// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Divider, Flex, Heading, Text } from '@geti/ui';

import { TrainModel } from '../train-model/train-model.component';
import { Header } from './components/header.component';
import { CurrentModelTraining } from './current-model-training/current-model-training.component';
import { ModelListing } from './model-listing.component';
import { ModelListingProvider, useModelListing } from './provider/model-listing-provider';

const ModelListingContent = () => {
    const { groupedModels, searchBy } = useModelListing();

    const hasNoResults = groupedModels.length === 0 && searchBy.length > 0;
    const hasNoModels = groupedModels.length === 0 && searchBy.length === 0;

    if (hasNoModels) {
        return (
            <Flex
                direction={'column'}
                height={'100%'}
                alignItems={'center'}
                justifyContent={'center'}
                UNSAFE_style={{ padding: dimensionValue('size-300') }}
            >
                <CurrentModelTraining />

                <Flex direction={'column'} alignItems={'center'} gap={'size-100'} marginTop={'size-600'}>
                    <Heading level={2}>No models yet. Train your first model to get started.</Heading>
                    <TrainModel />
                </Flex>
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
