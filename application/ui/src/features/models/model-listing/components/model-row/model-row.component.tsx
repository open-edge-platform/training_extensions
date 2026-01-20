// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex, Grid, Item, Key, Menu, MenuTrigger, Tag, Text } from '@geti/ui';
import { MoreMenu } from '@geti/ui/icons';

import { ReactComponent as ThumbsUp } from '../../../../../assets/icons/thumbs-up.svg';
import { Model } from '../../../../../constants/shared-types';
import { GRID_COLUMNS } from '../../constants';
import { AccuracyIndicator } from '../../model-variants/accuracy-indicator.component';
import { formatTrainingDateTime } from '../../utils/date-formatting';
import { formatModelSize } from '../../utils/format-model-size';
import { ActiveModelTag } from '../active-model-tag.component';
import { ParentRevisionModel } from '../parent-revision-model.component';

import styles from './model-row.module.scss';

type ModelRowProps = {
    model: Model;
    activeModelArchitectureId?: string;
    parentRevisionModel?: Model;
    onExpandModel?: (modelId: string) => void;
    onModelAction?: (key: Key) => void;
};

const getTotalModelSize = (model: Model): number => {
    return (model.variants ?? []).reduce((total, variant) => total + (variant.weights_size ?? 0), 0);
};

export const ModelRow = ({
    model,
    activeModelArchitectureId,
    parentRevisionModel,
    onExpandModel,
    onModelAction,
}: ModelRowProps) => {
    const trainingEndTime = model.training_info.end_time;
    const totalSize = getTotalModelSize(model);

    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} columnGap={'size-200'}>
            <Flex direction={'column'} gap={'size-50'}>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Text UNSAFE_className={styles.modelName}>{model.name ?? 'Unnamed Model'}</Text>
                    {model.id === activeModelArchitectureId && <ActiveModelTag />}
                </Flex>
                <Text UNSAFE_className={styles.secondaryText}>
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

            <Text UNSAFE_className={styles.dateText}>{formatTrainingDateTime(trainingEndTime)}</Text>

            <Flex alignItems={'start'} direction={'column'} gap={'size-100'}>
                <Text UNSAFE_className={styles.smallText}>{model.architecture}</Text>
                {/* TODO: Speed is hardcoded for now, once the backend is update we need to update this */}
                <Tag prefix={<ThumbsUp />} text={'Speed'} className={styles.recommendedForTag} />
            </Flex>

            <Text UNSAFE_className={styles.smallText}>{totalSize > 0 ? formatModelSize(totalSize) : '-'}</Text>

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
