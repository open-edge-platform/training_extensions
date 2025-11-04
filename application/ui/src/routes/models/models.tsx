// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Disclosure, DisclosurePanel, DisclosureTitle, Divider, View } from '@geti/ui';
import { DatasetHeader } from 'src/features/models/dataset-header.component';
import { ModelsHeader } from 'src/features/models/models-header.component';

export const Models = () => {
    return (
        <View padding={'size-300'}>
            <ModelsHeader />

            <Divider size={'S'} marginY={'size-300'} />

            <DatasetHeader />

            {/* Dataset element (model) */}
            <Disclosure isQuiet>
                <DisclosureTitle>Model Project #3</DisclosureTitle>
                <DisclosurePanel>Content</DisclosurePanel>
            </Disclosure>
        </View>
    );
};
