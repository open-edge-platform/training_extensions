// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, dimensionValue, Flex, Grid, Item, Key, Menu, MenuTrigger, Tag, Text } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { SchemaModelView } from '../../../../../api/openapi-spec';
import { ReactComponent as ThumbsUp } from '../../../../../assets/icons/thumbs-up.svg';
import { GRID_COLUMNS } from '../../constants';
import { AccuracyIndicator } from '../../model-variants/accuracy-indicator.component';
import { formatTrainingDateTime } from '../../utils/date-formatting';
import { ActiveModelTag } from '../active-model-tag.component';
import { ParentRevisionModel } from '../parent-revision-model.component';

type ModelRowProps = {
    model: SchemaModelView;
    activeModelArchitectureId?: string;
    parentRevisionModel?: SchemaModelView;
    onExpandModel?: (modelId: string) => void;
    onModelAction?: (key: Key) => void;
};

export const ModelRow = ({
    model,
    activeModelArchitectureId,
    parentRevisionModel,
    onExpandModel,
    onModelAction,
}: ModelRowProps) => {
    const trainingEndTime = model.training_info.end_time;

    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} columnGap={'size-200'}>
            <Flex direction={'column'} gap={'size-50'}>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-200'), fontWeight: '400' }}>
                        {model.name ?? 'Unnamed Model'}
                    </Text>
                    {model.id === activeModelArchitectureId && <ActiveModelTag />}
                </Flex>
                <Text
                    UNSAFE_style={{
                        fontSize: dimensionValue('font-size-75'),
                        color: 'var(--spectrum-global-color-gray-700)',
                    }}
                >
                    {parentRevisionModel ? (
                        <ParentRevisionModel
                            id={parentRevisionModel.id}
                            name={parentRevisionModel.name}
                            onExpandModel={onExpandModel}
                        />
                    ) : (
                        <div />
                    )}
                </Text>
            </Flex>

            <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75'), whiteSpace: 'pre-line' }}>
                {formatTrainingDateTime(trainingEndTime)}
            </Text>

            <Flex alignItems={'start'} direction={'column'} gap={'size-100'}>
                <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75'), fontWeight: '400' }}>
                    {model.architecture}
                </Text>
                {/* TODO: Speed is hardcoded for now, once the backend is update we need to update this */}
                <Tag
                    prefix={<ThumbsUp />}
                    text={'Speed'}
                    style={{
                        borderRadius: dimensionValue('size-50'),
                        backgroundColor: 'var(--spectrum-global-color-gray-300)',
                        color: 'var(--spectrum-global-color-gray-700)',
                        fontWeight: '400',
                    }}
                />
            </Flex>

            <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75'), fontWeight: '400' }}>500 MB</Text>

            <AccuracyIndicator accuracy={72} />

            {onModelAction ? (
                <MenuTrigger>
                    <ActionButton isQuiet>
                        <MoreMenu />
                    </ActionButton>
                    <Menu onAction={onModelAction}>
                        <Item key='rename'>Rename</Item>
                        <Item key='delete'>Delete</Item>
                        <Item key='export'>Export</Item>
                    </Menu>
                </MenuTrigger>
            ) : null}
        </Grid>
    );
};
