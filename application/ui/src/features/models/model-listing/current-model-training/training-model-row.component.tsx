// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Divider, Flex, Grid, Loading, Tag, Text } from '@geti/ui';
import dayjs from 'dayjs';
import duration from 'dayjs/plugin/duration';

import { ReactComponent as ThumbsUp } from '../../../../assets/icons/thumbs-up.svg';
import { Job } from '../../../../constants/shared-types';
import { GRID_COLUMNS } from '../constants';
import { BottomProgressBar } from './bottom-progress-bar.component';

import classes from './current-model-training.module.scss';

dayjs.extend(duration);

type TrainingModelRowProps = {
    job: Job;
    onCancel?: () => void;
};

const TrainingTag = () => (
    <Tag prefix={<Loading size={'S'} mode={'inline'} />} className={classes.trainingTag} text={'Training'} />
);

const StatusTag = ({ status }: { status: string }) => (
    <Tag className={classes.statusTag} withDot={false} text={status} />
);

export const TrainingModelRow = ({ job, onCancel }: TrainingModelRowProps) => {
    const modelName = job.metadata.model.id.slice(0, 5) || 'Unnamed Model';

    return (
        <BottomProgressBar progress={job.progress}>
            <Grid
                columns={GRID_COLUMNS}
                alignItems={'center'}
                width={'100%'}
                columnGap={'size-200'}
                UNSAFE_className={classes.grid}
            >
                <Flex direction={'column'} gap={'size-50'}>
                    <Flex alignItems={'center'}>
                        <Text UNSAFE_className={classes.modelName}>{modelName}</Text>
                        <TrainingTag />
                        <StatusTag status={job.message || 'running...'} />
                    </Flex>

                    <Text UNSAFE_className={classes.metaText}>
                        {`Started: ${dayjs(job.started_at).format('DD MMM YYYY, hh:mm A')}`}
                        <Divider orientation={'vertical'} />
                        {`Elapsed: ${dayjs.duration(dayjs().diff(dayjs(job.started_at))).format('mm:ss')}m`}
                    </Text>
                </Flex>

                <Text UNSAFE_className={classes.smallText}>...</Text>

                <Flex alignItems={'start'} direction={'column'} gap={'size-100'}>
                    <Text UNSAFE_className={classes.smallText}>{job.metadata.model.architecture}</Text>
                    {/* TODO: Speed is hardcoded for now, once the backend is update we need to update this */}
                    <Tag prefix={<ThumbsUp />} text={'Speed'} className={classes.recommendedForTag} />
                </Flex>

                <Text UNSAFE_className={classes.smallText}>...</Text>

                <Text UNSAFE_className={classes.smallText}>...</Text>

                {onCancel ? (
                    <Button
                        isDisabled={job.status !== 'running'}
                        variant={'negative'}
                        onPress={onCancel}
                        aria-label={'Cancel training job'}
                    >
                        Cancel
                    </Button>
                ) : (
                    <div />
                )}
            </Grid>
        </BottomProgressBar>
    );
};
