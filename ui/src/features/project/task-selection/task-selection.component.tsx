// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { Flex, Heading, Image, Radio, RadioGroup, Text, View } from '@geti/ui';

import thumbnailUrl from '../../../assets/mocked-project-thumbnail.png';
import { TaskOption, TaskType } from './interface';

import classes from './task-selection.module.scss';

const TASK_OPTIONS: TaskOption[] = [
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
        value: 'segmentation',
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
            <View maxWidth={'344px'}>
                <Image height={'size-3000'} width={'size-3600'} src={taskOption.imageSrc} alt={taskOption.title} />
            </View>

            <View padding={'size-300'} backgroundColor={'gray-100'}>
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

type TaskSelectionProps = { selectedTask: TaskType; setSelectedTask: Dispatch<SetStateAction<TaskType>> };
export const TaskSelection = ({ selectedTask, setSelectedTask }: TaskSelectionProps) => {
    const selectedTaskOption = TASK_OPTIONS.find((task) => task.value === selectedTask) || TASK_OPTIONS[0];

    return (
        <Flex direction={'column'} gap={'size-300'} alignItems={'center'}>
            <RadioGroup
                aria-label='Task selection'
                value={selectedTaskOption.value}
                onChange={(value: string) => {
                    const option = TASK_OPTIONS.find((taskOption) => taskOption.value === value);

                    if (option) setSelectedTask(option.value);
                }}
            >
                <Flex justifyContent={'center'} gap={'size-300'}>
                    {TASK_OPTIONS.map((taskOption) => (
                        <Option
                            key={taskOption.value}
                            taskOption={taskOption}
                            onPress={() => {
                                setSelectedTask(taskOption.value);
                            }}
                        />
                    ))}
                </Flex>
            </RadioGroup>

            <Flex>
                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>
                    {`What objects should the model learn to ${selectedTaskOption.verb}?`}
                </Text>
            </Flex>
        </Flex>
    );
};
