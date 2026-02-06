// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Badge, Flex, Grid, Text } from '@geti/ui';

import type { DatasetRevision, Model } from '../../../../../constants/shared-types';
import { GRID_COLUMNS } from '../../constants';
import { AccuracyIndicator } from '../../model-variants/accuracy-indicator.component';
import { type GroupByMode } from '../../types';
import { formatTrainingDateTime } from '../../utils/date-formatting';
import { formatModelSize } from '../../utils/format-model-size';
import { isFailedModel } from '../../utils/utils';
import { ActiveModelTag } from '../active-model-tag.component';
import { ParentRevisionModel } from '../parent-revision-model.component';
import { ArchitectureColumn } from './architecture-column.component';
import { DatasetColumn } from './dataset-revision-column.component';

import styles from './model-row.module.scss';

type ModelRowProps = {
    model: Model;
    activeModelId?: string;
    parentRevisionModel?: Model;
    onExpandModel?: (modelId: string) => void;
    groupBy: GroupByMode;
    datasetRevision: DatasetRevision | undefined;
};

const FailedModel = () => {
    return <Badge variant={'negative'}>Failed</Badge>;
};

export const ModelRow = ({
    model,
    activeModelId,
    parentRevisionModel,
    onExpandModel,
    groupBy,
    datasetRevision,
}: ModelRowProps) => {
    const trainingEndTime = model.training_info.end_time;
    const totalSize = model.size;
    const labelSchemaRevision = model.training_info.label_schema_revision ?? {};
    const labelsCount =
        'labels' in labelSchemaRevision && Array.isArray(labelSchemaRevision.labels)
            ? labelSchemaRevision.labels.length
            : undefined;

    return (
        <Grid columns={GRID_COLUMNS} alignItems={'center'} width={'100%'} columnGap={'size-200'}>
            <Flex direction={'column'} gap={'size-50'}>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Text UNSAFE_className={styles.modelName} data-testid={'model-name'}>
                        {model.name ?? 'Unnamed Model'}
                        {isFailedModel(model) && (
                            <>
                                {' '}
                                <FailedModel />
                            </>
                        )}
                    </Text>
                    {model.id === activeModelId && <ActiveModelTag />}
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

            {groupBy === 'architecture' ? (
                <DatasetColumn datasetRevision={datasetRevision} labelsCount={labelsCount} />
            ) : (
                <ArchitectureColumn architecture={model.architecture} />
            )}

            <Text UNSAFE_className={styles.smallText}>{totalSize > 0 ? formatModelSize(totalSize) : '-'}</Text>

            <AccuracyIndicator accuracy={72} />
        </Grid>
    );
};
