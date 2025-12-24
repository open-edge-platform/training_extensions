// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    ActionButton,
    Disclosure,
    DisclosurePanel,
    DisclosureTitle,
    Divider,
    Flex,
    Grid,
    Item,
    Menu,
    MenuTrigger,
    Tag,
    Text,
    View,
} from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { ReactComponent as StartIcon } from '../../../assets/icons/start.svg';
import { DatasetHeader } from './dataset-header.component';
import { AccuracyIndicator } from './model-variants/accuracy-indicator.component';
import { ModelVariants } from './model-variants/model-variants.component';
import { ModelsHeader } from './models-header.component';

import classes from './model-listing.module.scss';

// TODO: Replace with dynamic data
const models = [
    { id: 1, name: 'Model Project #1' },
    { id: 2, name: 'Model Project #2' },
];

const GRID_COLUMNS = ['2fr 1fr 1fr 1fr 1fr auto'];

const HeaderRow = () => {
    return (
        <Grid
            columns={GRID_COLUMNS}
            gap={'size-200'}
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
            <div />
        </Grid>
    );
};

// TODO: Update model interface
const ModelVariantItem = ({ model }: { model: { id: number; name: string } }) => {
    return (
        <Grid columns={GRID_COLUMNS} gap={'size-200'} alignItems={'center'} width={'100%'}>
            <Flex direction={'column'} gap={'size-50'}>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Text UNSAFE_style={{ fontSize: 'var(--spectrum-global-dimension-font-size-75)' }}>
                        {model.name}
                    </Text>
                    <Tag
                        prefix={<StartIcon />}
                        style={{
                            backgroundColor: 'var(--energy-blue)',
                            color: 'var(--spectrum-global-color-gray-50)',
                            borderRadius: 'var(--spectrum-global-dimension-size-50)',
                            padding:
                                'var(--spectrum-global-dimension-size-25) var(--spectrum-global-dimension-size-50)',
                        }}
                        text={'Active'}
                    />
                </Flex>
                <Text
                    UNSAFE_style={{
                        fontSize: 'var(--spectrum-global-dimension-font-size-75)',
                        color: 'var(--spectrum-global-color-gray-700)',
                    }}
                >
                    Fine-tuned from Model Project #1
                </Text>
            </Flex>

            <Flex direction={'column'} gap={'size-25'}>
                <Text UNSAFE_style={{ fontSize: 'var(--spectrum-global-dimension-font-size-75)' }}>01 Oct 2025</Text>
                <Text
                    UNSAFE_style={{
                        fontSize: 'var(--spectrum-global-dimension-font-size-75)',
                        color: 'var(--spectrum-global-color-gray-700)',
                    }}
                >
                    11:07 AM
                </Text>
            </Flex>

            <Text UNSAFE_style={{ fontSize: 'var(--spectrum-global-dimension-font-size-75)' }}>YOLOX-S</Text>

            <Text UNSAFE_style={{ fontSize: 'var(--spectrum-global-dimension-font-size-75)' }}>500 MB</Text>

            <AccuracyIndicator accuracy={72} />

            <MenuTrigger onOpenChange={() => {}}>
                <ActionButton isQuiet>
                    <MoreMenu />
                </ActionButton>
                <Menu>
                    <Item key='delete'>Delete</Item>
                    <Item key='export'>Export</Item>
                </Menu>
            </MenuTrigger>
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
                        <DisclosureTitle UNSAFE_className={classes.disclosureItem}>
                            <ModelVariantItem model={model} />
                        </DisclosureTitle>
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
