// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Dispatch, SetStateAction } from 'react';

import { Divider, Flex, Grid, Heading, Image, Radio, RadioGroup, Text, View } from '@geti/ui';

import classificationImageUrl from '../../../assets/classification.webp';
import detectionImageUrl from '../../../assets/detection.webp';
import segmentationImageUrl from '../../../assets/segmentation.webp';
import type { TaskType } from '../../../constants/shared-types';
import type { TaskOption } from './interface';

import classes from './task-selection.module.scss';

export const TASK_OPTIONS: TaskOption[] = [
    {
        id: 'detection_task',
        imageSrc: detectionImageUrl,
        title: 'Object Detection',
        description: 'Identify and locate objects in your images',
        advice: 'Best for: Counting, Tracking',
        verb: 'detect',
        value: 'detection',
    },
    {
        id: 'segmentation_task',
        imageSrc: segmentationImageUrl,
        title: 'Instance Segmentation',
        description: 'Detect and outline specific regions or shapes',
        advice: 'Best for: Measurement, Odd shapes',
        verb: 'segment',
        value: 'instance_segmentation',
    },
    {
        id: 'classification_task',
        imageSrc: classificationImageUrl,
        title: 'Image Classification',
        description: 'Categorize entire images based on their content',
        advice: 'Best for: Filtering, Content Moderation',
        verb: 'classify',
        value: 'classification',
    },
];

type TaskOptionProps = {
    taskOption: TaskOption;
    onPress: () => void;
};

const Option = ({ taskOption, onPress }: TaskOptionProps) => {
    return (
        <div onClick={onPress} className={classes.option} aria-label={`Task option: ${taskOption.title}`}>
            <View>
                <Image height={'size-2400'} width={'100%'} src={taskOption.imageSrc} alt={taskOption.title} />
            </View>

            <View padding={'size-200'}>
                <Flex justifyContent={'space-between'} gap={'size-50'} alignItems={'center'}>
                    <Heading level={2} UNSAFE_className={classes.title}>
                        {taskOption.title}
                    </Heading>
                    <Radio aria-label={taskOption.value} value={taskOption.value} />
                </Flex>

                <Text UNSAFE_className={classes.description}>{taskOption.description}</Text>

                <Divider marginTop={'size-100'} marginBottom={'size-150'} size={'S'} />

                <Text>{taskOption.advice}</Text>
            </View>
        </div>
    );
};

type TaskSelectionProps = { selectedTask: TaskType | null; setSelectedTask: Dispatch<SetStateAction<TaskType | null>> };

export const TaskSelection = ({ selectedTask, setSelectedTask }: TaskSelectionProps) => {
    const selectedTaskOption = TASK_OPTIONS.find((task) => task.value === selectedTask);

    return (
        <Flex direction={'column'} gap={'size-300'} alignItems={'center'}>
            <RadioGroup
                aria-label='Task selection'
                width={'100%'}
                value={selectedTaskOption?.value}
                onChange={(value: string) => {
                    const option = TASK_OPTIONS.find((taskOption) => taskOption.value === value);

                    if (option) setSelectedTask(option.value);
                }}
            >
                <Grid
                    columns={
                        'repeat(3, minmax(min(100%, var(--spectrum-global-dimension-size-3600)), ' +
                        'var(--spectrum-global-dimension-size-4600)))'
                    }
                    gap={'size-300'}
                    width={'100%'}
                    justifyContent={'center'}
                >
                    {TASK_OPTIONS.map((taskOption) => (
                        <Option
                            key={taskOption.value}
                            taskOption={taskOption}
                            onPress={() => {
                                setSelectedTask(taskOption.value);
                            }}
                        />
                    ))}
                </Grid>
            </RadioGroup>
        </Flex>
    );
};
