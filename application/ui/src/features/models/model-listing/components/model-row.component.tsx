// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, dimensionValue, Flex, Grid, Item, Menu, MenuTrigger, Text } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { SchemaModelView } from '../../../../api/openapi-spec';
import { useGetModel } from '../../hooks/api/use-get-model.hook';
import { GRID_COLUMNS } from '../constants';
import { AccuracyIndicator } from '../model-variants/accuracy-indicator.component';
import { useModelListing } from '../provider/model-listing-provider';
import { formatTrainingDateTime } from '../utils/date-formatting';
import { ActiveModelTag } from './active-model-tag.component';
import { ParentRevisionModel } from './parent-revision-model.component';

interface ModelRowProps {
    model: SchemaModelView;
}

export const ModelRow = ({ model }: ModelRowProps) => {
    const { activeModelId, onExpandModel } = useModelListing();

    const trainingEndTime = model.training_info.end_time;
    const parentRevisionModel = useGetModel(model.parent_revision);

    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'}>
            <Flex direction={'column'} gap={'size-50'}>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-200') }}>
                        {model.name ?? 'Unnamed Model'}
                    </Text>
                    {model.id === activeModelId && <ActiveModelTag />}
                </Flex>
                <Text
                    UNSAFE_style={{
                        fontSize: dimensionValue('font-size-75'),
                        color: 'var(--spectrum-global-color-gray-700)',
                    }}
                >
                    {parentRevisionModel?.data ? (
                        <ParentRevisionModel
                            id={parentRevisionModel.data.id}
                            name={parentRevisionModel.data.name}
                            onExpandModel={onExpandModel}
                        />
                    ) : null}
                </Text>
            </Flex>

            <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75'), whiteSpace: 'pre-line' }}>
                {formatTrainingDateTime(trainingEndTime)}
            </Text>

            <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75') }}>{model.architecture}</Text>

            <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75') }}>500 MB</Text>

            <AccuracyIndicator accuracy={72} />

            <MenuTrigger>
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
