// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Divider, Flex, Form, Text } from '@geti/ui';
import { useNavigate } from 'react-router';
import { v4 as uuid } from 'uuid';

import { $api } from '../../../api/client';
import { paths } from '../../../constants/paths';
import { Label } from '../../annotator/types';
import { LabelSelection } from '../label-selection/label-selection.component';
import { TaskType } from '../task-selection/interface';
import { TaskSelection } from '../task-selection/task-selection.component';
import { ProjectName } from './project-name.component';

import classes from './create-project-form.module.scss';

export const CreateProjectForm = () => {
    const [selectedTask, setSelectedTask] = useState<TaskType>('detection');
    const [labels, setLabels] = useState<Label[]>([{ id: uuid(), color: '#F20004', name: 'Object' }]);
    const [name, setName] = useState<string>('Project #1');

    const navigate = useNavigate();
    const createProjectMutation = $api.useMutation('post', '/api/projects');

    const createProject = (e: FormEvent) => {
        e.preventDefault();

        const projectId = uuid();

        createProjectMutation.mutate(
            {
                body: {
                    id: projectId,
                    task: {
                        task_type: selectedTask,
                        exclusive_labels: selectedTask === 'classification',
                        labels: labels.map((label) => ({ name: label.name })),
                    },
                    name,
                },
            },
            {
                onSuccess: () => {
                    navigate(paths.project.inference({ projectId }));
                },
            }
        );
    };

    return (
        <Form onSubmit={createProject} maxWidth={'1052px'} margin={'0 auto'}>
            <Flex
                direction={'column'}
                gap='size-600'
                alignItems={'center'}
                marginTop={'size-1000'}
                marginBottom={'size-400'}
            >
                <ProjectName name={name} setName={setName} />

                <Text
                    UNSAFE_style={{
                        color: 'var(--spectrum-global-color-gray-700)',
                        textAlign: 'center',
                    }}
                >
                    What type of task would you like the model to perform?
                </Text>
            </Flex>

            <Flex direction='column' gap='size-300' UNSAFE_style={{ overflow: 'auto', margin: '0 auto' }}>
                <TaskSelection selectedTask={selectedTask} setSelectedTask={setSelectedTask} />

                <Divider size='S' />

                <LabelSelection labels={labels} setLabels={setLabels} />
            </Flex>

            <Flex justifyContent={'end'} UNSAFE_className={classes.buttonGroup}>
                <ButtonGroup>
                    <Button type={'submit'} variant='accent'>
                        Create project
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};
