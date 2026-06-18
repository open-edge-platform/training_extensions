// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { AlertDialog, Button, DialogContainer, Flex, Grid, Loading, Tag, Text } from '@geti/ui';
import dayjs from 'dayjs';
import duration from 'dayjs/plugin/duration';
import { useStreamJobStatus } from 'hooks/api/jobs/jobs.hook';
import { isTrainJob } from 'hooks/api/util';
import { capitalize } from 'lodash-es';

import { DatasetRevision, Job, ModelArchitectureWithPerformanceCategory } from '../../../../constants/shared-types';
import { useGetModel } from '../../hooks/api/use-get-model.hook';
import { TrainingLogsDialog } from '../../training-logs/training-logs-dialog.component';
import { ArchitectureColumn } from '../components/model-row/architecture-column.component';
import { DatasetColumn } from '../components/model-row/dataset-revision-column.component';
import { GroupByMode } from '../types';
import { BottomProgressBar } from './bottom-progress-bar.component';
import { RUNNING_JOB_GRID_COLUMNS } from './running-job-table-header.component';

import classes from './current-model-running.module.scss';

dayjs.extend(duration);

type RunningModelRowProps = {
    job: Job;
    onCancel?: () => void;
    groupBy: GroupByMode;
    datasetRevisions: DatasetRevision[];
    modelArchitectures: ModelArchitectureWithPerformanceCategory[];
};

const StatusTag = ({ status }: { status: string }) => (
    <Tag prefix={<Loading size={'S'} mode={'inline'} />} className={classes.runningStatusTag} text={status} />
);

const StatusTagMessage = ({ status }: { status: string }) => (
    <Tag className={classes.statusTag} withDot={false} text={status} />
);

type CancelRunningJobProps = {
    job: Job;
    onCancel: () => void;
};

const CancelRunningJob = ({ job, onCancel }: CancelRunningJobProps) => {
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState<boolean>(false);

    return (
        <>
            <Button
                isDisabled={job.status !== 'RUNNING' && job.status !== 'PENDING'}
                variant={'negative'}
                onPress={() => setIsDeleteDialogOpen(true)}
                aria-label={'Cancel job'}
            >
                Cancel
            </Button>
            <DialogContainer onDismiss={() => setIsDeleteDialogOpen(false)}>
                {isDeleteDialogOpen && (
                    <AlertDialog
                        title='Stop job'
                        variant='destructive'
                        primaryActionLabel='Cancel'
                        onPrimaryAction={onCancel}
                        cancelLabel='Close'
                    >
                        Are you sure you want to stop this job?
                    </AlertDialog>
                )}
            </DialogContainer>
        </>
    );
};

const ViewLogsButton = ({ jobId }: { jobId: string }) => {
    const [isLogsDialogOpen, setIsLogsDialogOpen] = useState(false);

    return (
        <>
            <Button variant={'secondary'} onPress={() => setIsLogsDialogOpen(true)} aria-label={'View logs'}>
                Logs
            </Button>
            <DialogContainer type={'fullscreen'} onDismiss={() => setIsLogsDialogOpen(false)}>
                {isLogsDialogOpen && <TrainingLogsDialog jobId={jobId} />}
            </DialogContainer>
        </>
    );
};

export const RunningModelRow = ({
    job,
    onCancel,
    datasetRevisions,
    groupBy,
    modelArchitectures,
}: RunningModelRowProps) => {
    useStreamJobStatus(job.job_id);

    const modelId = 'model' in job.metadata ? job.metadata.model?.id : undefined;
    const { data: trainingModel } = useGetModel(modelId);

    const device = isTrainJob(job) ? job.metadata.device.name : null;

    const modelArchitectureId =
        'model' in job.metadata && 'architecture' in job.metadata.model && job.metadata.model.architecture;
    const modelName = trainingModel?.name;

    const modelArchitecture = modelArchitectures.find(({ id }) => id === modelArchitectureId);

    const datasetRevision = datasetRevisions.find(({ id }) => id === trainingModel?.training_info.dataset_revision_id);
    const labelSchemaRevision = trainingModel?.training_info.label_schema_revision ?? {};
    const labelsCount =
        'labels' in labelSchemaRevision && Array.isArray(labelSchemaRevision.labels)
            ? labelSchemaRevision.labels.length
            : undefined;

    const formattedStartedAt = job.started_at
        ? dayjs(job.started_at).format('DD MMM YYYY, hh:mm A')
        : 'Waiting to start...';

    const statusMessage = job.message || (job.status === 'PENDING' ? 'Pending...' : 'Running...');

    const showStatusTagMessage =
        job.status.toLocaleLowerCase() !== statusMessage.replace('...', '').toLocaleLowerCase();

    return (
        <BottomProgressBar progress={job.progress}>
            <Grid
                columns={RUNNING_JOB_GRID_COLUMNS}
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
                        <StatusTag status={capitalize(job.status)} />
                        {showStatusTagMessage && <StatusTagMessage status={statusMessage} />}
                    </Flex>

                    <Text UNSAFE_className={classes.metaText}>{`Started: ${formattedStartedAt}`}</Text>
                    {device && <Text UNSAFE_className={classes.metaText}>{`Device: ${device}`}</Text>}
                </Flex>

                <Flex alignItems={'start'} direction={'column'} gap={'size-100'}>
                    {groupBy === 'architecture' ? (
                        <DatasetColumn datasetRevision={datasetRevision} labelsCount={labelsCount} />
                    ) : (
                        <ArchitectureColumn architecture={modelArchitecture} />
                    )}
                </Flex>

                <Flex gap={'size-100'} direction={'column'} alignItems={'center'}>
                    <ViewLogsButton jobId={job.job_id} />
                    {onCancel ? <CancelRunningJob onCancel={onCancel} job={job} /> : null}
                </Flex>
            </Grid>
        </BottomProgressBar>
    );
};
