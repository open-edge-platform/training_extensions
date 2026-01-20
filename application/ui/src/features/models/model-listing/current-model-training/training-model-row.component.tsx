// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Divider, Flex, Grid, Loading, Tag, Text } from '@geti/ui';
import dayjs from 'dayjs';
import duration from 'dayjs/plugin/duration';

import { ReactComponent as ThumbsUp } from '../../../../assets/icons/thumbs-up.svg';
import { Job } from '../../../../constants/shared-types';
import { GRID_COLUMNS } from '../constants';
import { BottomProgressBar } from './bottom-progress-bar.component';

dayjs.extend(duration);

type TrainingModelRowProps = {
    job: Job;
    onCancel?: () => void;
};

const TrainingTag = () => (
    <Tag
        prefix={<Loading size={'S'} mode={'inline'} />}
        style={{
            backgroundColor: 'var(--energy-blue)',
            color: 'var(--spectrum-global-color-gray-50)',
            borderRadius: dimensionValue('size-50'),
            borderTopRightRadius: 0,
            borderBottomRightRadius: 0,
            padding: `${dimensionValue('size-25')} ${dimensionValue('size-50')}`,
        }}
        text={'Training'}
    />
);

const StatusTag = ({ status }: { status: string }) => (
    <Tag
        style={{
            backgroundColor: 'var(--energy-blue-shade-2)',
            color: 'var(--spectrum-global-color-gray-800)',
            borderRadius: dimensionValue('size-50'),
            borderTopLeftRadius: 0,
            borderBottomLeftRadius: 0,
            padding: `${dimensionValue('size-25')} ${dimensionValue('size-50')}`,
        }}
        withDot={false}
        text={status}
    />
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
                UNSAFE_style={{
                    padding: `${dimensionValue('size-150')} ${dimensionValue('size-250')}
                        ${dimensionValue('size-150')} ${dimensionValue('size-1000')}`,
                }}
            >
                <Flex direction={'column'} gap={'size-50'}>
                    <Flex alignItems={'center'}>
                        <Text
                            UNSAFE_style={{
                                fontSize: dimensionValue('font-size-200'),
                                paddingRight: dimensionValue('size-100'),
                            }}
                        >
                            {modelName}
                        </Text>
                        <TrainingTag />
                        <StatusTag status={job.message || 'running...'} />
                    </Flex>

                    <Text
                        UNSAFE_style={{
                            fontSize: dimensionValue('font-size-75'),
                            color: 'var(--spectrum-global-color-gray-700)',
                        }}
                    >
                        {`Started: ${dayjs(job.started_at).format('DD MMM YYYY, hh:mm A')}`}
                        <Divider orientation={'vertical'} />
                        {`Elapsed: ${dayjs.duration(dayjs().diff(dayjs(job.started_at))).format('mm:ss')}m`}
                    </Text>
                </Flex>

                <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75') }}>...</Text>

                <Flex alignItems={'start'} direction={'column'} gap={'size-100'}>
                    <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75') }}>
                        {job.metadata.model.architecture}
                    </Text>
                    {/* TODO: Speed is hardcoded for now, once the backend is update we need to update this */}
                    <Tag
                        prefix={<ThumbsUp />}
                        text={'Speed'}
                        style={{
                            borderRadius: dimensionValue('size-50'),
                            backgroundColor: 'var(--spectrum-global-color-gray-300)',
                            color: 'var(--spectrum-global-color-gray-700)',
                        }}
                    />
                </Flex>

                <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75') }}>...</Text>

                <Text UNSAFE_style={{ fontSize: dimensionValue('font-size-75') }}>...</Text>

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
