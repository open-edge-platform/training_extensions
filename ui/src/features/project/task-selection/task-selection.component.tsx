// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Flex, Heading, Image, Radio, RadioGroup, Text, View } from '@geti/ui';

import thumbnailUrl from '../../../assets/mocked-project-thumbnail.png';

import classes from './task-selection.module.scss';

type TaskOption = {
    id: string;
    imageSrc: string;
    title: string;
    description: string;
    verb: string;
    value: string;
};
const TASKS: TaskOption[] = [
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
    task: TaskOption;
    onPress: () => void;
};
const TaskOption = ({ task, onPress }: TaskOptionProps) => {
    return (
        <div onClick={onPress} className={classes.option} aria-label={`Task option: ${task.title}`}>
            <View maxWidth={'344px'}>
                <Image height={'size-3000'} width={'size-3600'} src={task.imageSrc} alt={task.title} />
            </View>

            <View padding={'size-300'} backgroundColor={'gray-100'}>
                <Flex justifyContent={'space-between'} alignItems={'center'}>
                    <Heading level={2} UNSAFE_className={classes.title}>
                        {task.title}
                    </Heading>
                    <Radio aria-label={task.value} value={task.value} />
                </Flex>

                <Text UNSAFE_className={classes.description}>{task.description}</Text>
            </View>
        </div>
    );
};

export const TaskSelection = () => {
    const [selectedTask, setSelectedTask] = useState(TASKS[0]);

    return (
        <Flex direction={'column'} gap={'size-300'} alignItems={'center'}>
            <RadioGroup
                aria-label='Task selection'
                value={selectedTask.value}
                onChange={(value: string) => {
                    const option = TASKS.find((task) => task.value === value);

                    if (option) setSelectedTask(option);
                }}
            >
                <Flex justifyContent={'center'} gap={'size-300'}>
                    {TASKS.map((task) => (
                        <TaskOption
                            key={task.value}
                            task={task}
                            onPress={() => {
                                setSelectedTask(task);
                            }}
                        />
                    ))}
                </Flex>
            </RadioGroup>

            <Flex>
                <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-700)' }}>
                    {`What objects should the model learn to ${selectedTask.verb}?`}
                </Text>
            </Flex>
        </Flex>
    );
};
