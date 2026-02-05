// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { AlertDialog, Button, DialogContainer, Divider, Flex, Grid, Loading, Tag, Text } from '@geti/ui';
import dayjs from 'dayjs';
import duration from 'dayjs/plugin/duration';

import type { DatasetRevision, Job } from '../../../../constants/shared-types';
import { useGetModel } from '../../hooks/api/use-get-model.hook';
import { ArchitectureColumn } from '../components/model-row/architecture-column.component';
import { DatasetColumn } from '../components/model-row/dataset-revision-column.component';
import { GRID_COLUMNS } from '../constants';
import { GroupByMode } from '../types';
import { BottomProgressBar } from './bottom-progress-bar.component';

import classes from './current-model-training.module.scss';

dayjs.extend(duration);

type TrainingModelRowProps = {
    job: Job;
    onCancel?: () => void;
    groupBy: GroupByMode;
    datasetRevisions: DatasetRevision[];
};

const TrainingTag = () => (
    <Tag prefix={<Loading size={'S'} mode={'inline'} />} className={classes.trainingTag} text={'Training'} />
);

const StatusTag = ({ status }: { status: string }) => (
    <Tag className={classes.statusTag} withDot={false} text={status} />
);

type CancelTrainingProps = {
    job: Job;
    onCancel: () => void;
};

const CancelTraining = ({ job, onCancel }: CancelTrainingProps) => {
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);

    return (
        <>
            <Button
                isDisabled={job.status !== 'RUNNING'}
                variant={'negative'}
                onPress={() => setIsDeleteDialogOpen(true)}
                aria-label={'Cancel training job'}
            >
                Cancel
            </Button>
            <DialogContainer onDismiss={() => setIsDeleteDialogOpen(false)}>
                {isDeleteDialogOpen && (
                    <AlertDialog
                        title='Cancel training'
                        variant='destructive'
                        primaryActionLabel='Cancel'
                        onPrimaryAction={onCancel}
                        cancelLabel='Close'
                    >
                        Are you sure you want to cancel training job?
                    </AlertDialog>
                )}
            </DialogContainer>
        </>
    );
};

export const TrainingModelRow = ({ job, onCancel, datasetRevisions, groupBy }: TrainingModelRowProps) => {
    const modelId = 'model' in job.metadata ? job.metadata.model?.id : undefined;
    const { data: trainingModel } = useGetModel(modelId);
    const modelArchitecture = 'model' in job.metadata && job.metadata.model?.architecture;
    const modelName = trainingModel?.name || modelId;

    const datasetRevision = datasetRevisions.find(({ id }) => id === trainingModel?.training_info.dataset_revision_id);
    const labelSchemaRevision = trainingModel?.training_info.label_schema_revision ?? {};
    const labelsCount =
        'labels' in labelSchemaRevision && Array.isArray(labelSchemaRevision.labels)
            ? labelSchemaRevision.labels.length
            : undefined;

    return (
        <BottomProgressBar progress={job.progress}>
            <Grid
                columns={GRID_COLUMNS}
                alignItems={'center'}
                width={'100%'}
                columnGap={'size-200'}
                UNSAFE_className={classes.grid}
            >
                <Flex direction={'column'} justifyContent={'center'} gap={'size-50'}>
                    <Flex alignItems={'center'}>
                        <Text UNSAFE_className={classes.modelName}>{modelName}</Text>
                    </Flex>

                    <Flex alignItems={'start'}>
                        <TrainingTag />
                        <StatusTag status={job.message || 'Running...'} />
                    </Flex>

                    <Text UNSAFE_className={classes.metaText}>
                        {`Started: ${dayjs(job.started_at).format('DD MMM YYYY, hh:mm A')}`}
                        <Divider orientation={'vertical'} />
                        {`Elapsed: ${dayjs.duration(dayjs().diff(dayjs(job.started_at))).format('mm:ss')}m`}
                    </Text>
                </Flex>

                <Text UNSAFE_className={classes.smallText}>...</Text>

                <Flex alignItems={'start'} direction={'column'} gap={'size-100'}>
                    {groupBy === 'architecture' ? (
                        <DatasetColumn datasetRevision={datasetRevision} labelsCount={labelsCount} />
                    ) : (
                        <ArchitectureColumn architecture={String(modelArchitecture)} />
                    )}
                </Flex>

                <Text UNSAFE_className={classes.smallText}>...</Text>

                <Text UNSAFE_className={classes.smallText}>...</Text>

                {onCancel ? <CancelTraining onCancel={onCancel} job={job} /> : <div />}
            </Grid>
        </BottomProgressBar>
    );
};
