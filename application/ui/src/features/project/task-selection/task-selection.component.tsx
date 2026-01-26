// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Dispatch, SetStateAction } from 'react';

import { Flex, Grid, Heading, Image, Radio, RadioGroup, Text, View } from '@geti/ui';

import thumbnailUrl from '../../../assets/mocked-project-thumbnail.png';
import { TaskType } from '../../../constants/shared-types';
import type { TaskOption } from './interface';

import classes from './task-selection.module.scss';

export const TASK_OPTIONS: TaskOption[] = [
    {
        id: 'detection_task',
        imageSrc: thumbnailUrl,
        title: 'Object Detection',
        description: 'Identify and locate objects in your images',
        verb: 'detect',
        value: 'detection',
    },
    {
        id: 'segmentation_task',
        imageSrc: thumbnailUrl,
        title: 'Image Segmentation',
        description: 'Detect and outline specific regions or shapes',
        verb: 'segment',
        value: 'instance_segmentation',
    },
    {
        id: 'classification_task',
        imageSrc: thumbnailUrl,
        title: 'Image Classification',
        description: 'Categorize entire images based on their content',
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

            <View padding={'size-300'}>
                <Flex justifyContent={'space-between'} alignItems={'center'}>
                    <Heading level={2} UNSAFE_className={classes.title}>
                        {taskOption.title}
                    </Heading>
                    <Radio aria-label={taskOption.value} value={taskOption.value} />
                </Flex>

                <Text UNSAFE_className={classes.description}>{taskOption.description}</Text>
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
                        'repeat(3, minmax(min(100%, var(--spectrum-global-dimension-size-3000)), ' +
                        'var(--spectrum-global-dimension-size-4600)))'
                    }
                    gap={'size-500'}
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
