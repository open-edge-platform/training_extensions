// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Disclosure, DisclosurePanel, DisclosureTitle, Divider, Grid, repeat, Text, View } from '@geti/ui';

import { DatasetHeader } from './dataset-header.component';
import { ModelVariants } from './model-variants/model-variants.component';
import { ModelsHeader } from './models-header.component';

import classes from './model-listing.module.scss';

const models = [
    { id: 1, name: 'Model Project #1' },
    { id: 2, name: 'Model Project #2' },
];

const HeaderRow = () => {
    return (
        <Grid
            columns={['2fr 1fr 1fr 1fr 1fr']}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
                padding: 'var(--spectrum-global-dimension-size-150) var(--spectrum-global-dimension-size-1000)',
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
                    <Disclosure key={model.id} isQuiet UNSAFE_className={classes.disclosure}>
                        <DisclosureTitle>{model.name}</DisclosureTitle>
                        <DisclosurePanel>
                            <ModelVariants />
                            {/* Model metrics */}
                            {/* Training parameter settings */}
                            {/* Training datasets */}
                        </DisclosurePanel>
                    </Disclosure>
                ))}
            </View>
        </View>
    );
};
