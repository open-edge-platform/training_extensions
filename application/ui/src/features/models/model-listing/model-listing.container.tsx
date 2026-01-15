// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ModelListing } from './model-listing.component';
import { ModelListingProvider } from './provider/model-listing-provider';

export const ModelListingContainer = () => {
    return (
        <ModelListingProvider>
            <ModelListing />
        </ModelListingProvider>
    );
};
