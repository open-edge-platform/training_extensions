// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex } from '@geti/ui';

import { Header } from './components/header.component';
import { CurrentModelTraining } from './current-model-training/current-model-training.component';
import { ModelListing } from './model-listing.component';
import { ModelListingProvider } from './provider/model-listing-provider';

export const ModelListingContainer = () => {
    return (
        <ModelListingProvider>
            <Flex
                direction={'column'}
                height={'100%'}
                UNSAFE_style={{ padding: 'var(--spectrum-global-dimension-size-300)' }}
            >
                <Header />

                <Divider size={'S'} marginY={'size-300'} />

                <CurrentModelTraining />

                <ModelListing />
            </Flex>
        </ModelListingProvider>
    );
};
