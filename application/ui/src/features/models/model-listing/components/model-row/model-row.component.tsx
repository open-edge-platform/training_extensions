// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Tag, Text } from '@geti/ui';

import { ReactComponent as ThumbsUp } from '../../../../../assets/icons/thumbs-up.svg';
import type { Model } from '../../../../../constants/shared-types';
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
};

export const ModelRow = ({ model, activeModelArchitectureId, parentRevisionModel, onExpandModel }: ModelRowProps) => {
    const trainingEndTime = model.training_info.end_time;
    const totalSize = model.size;

    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} columnGap={'size-200'}>
            <Flex direction={'column'} gap={'size-50'}>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Text UNSAFE_className={styles.modelName} data-testid={'model-name'}>
                        {model.name ?? 'Unnamed Model'}
                    </Text>
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
                <Text UNSAFE_className={styles.smallText}>{model.architecture} (Apache 2.0)</Text>
                {/* TODO: Speed is hardcoded for now, once the backend is update we need to update this */}
                <Tag prefix={<ThumbsUp />} text={'Speed'} className={styles.recommendedForTag} />
            </Flex>

            <Text UNSAFE_className={styles.smallText}>{totalSize > 0 ? formatModelSize(totalSize) : '-'}</Text>

            <AccuracyIndicator accuracy={72} />
        </Grid>
    );
};
