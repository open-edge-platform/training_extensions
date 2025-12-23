// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Disclosure, DisclosurePanel, DisclosureTitle, Divider, Grid, repeat, Text, View } from '@geti/ui';

import { DatasetHeader } from './dataset-header.component';
import { ModelVariants } from './model-variants/model-variants.component';
import { ModelsHeader } from './models-header.component';

const models = [
    { id: 1, name: 'Model Project #1' },
    { id: 2, name: 'Model Project #2' },
];

const HeaderRow = () => {
    return (
        <Grid
            gap='size-100'
            columns={repeat(5, '1fr')}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
                padding: 'var(--spectrum-global-dimension-size-150) var(--spectrum-global-dimension-size-300)',
            }}
        >
            <Text>Model Name</Text>
            <Text>Trained</Text>
            <Text>Architecture</Text>
            <Text>Total size</Text>
            <Text>Score</Text>
        </Grid>
    );
};

export const ModelListing = () => {
    return (
        <View padding={'size-300'} minWidth={0}>
            <ModelsHeader />

            <Divider size={'S'} marginY={'size-300'} />

            <DatasetHeader />

            <HeaderRow />

            <View>
                {models.map((model) => (
                    <Disclosure key={model.id}>
                        <DisclosureTitle>{model.name}</DisclosureTitle>
                        <DisclosurePanel>
                            <ModelVariants />
                        </DisclosurePanel>
                    </Disclosure>
                ))}
            </View>
        </View>
    );
};
