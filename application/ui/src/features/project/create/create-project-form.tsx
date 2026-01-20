// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Flex, Form, Text, TextField } from '@geti/ui';
import { useNavigate } from 'react-router';
import { v4 as uuid } from 'uuid';

import { paths } from '../../../constants/paths';
import type { Label, Project } from '../../../constants/shared-types';
import { useCreateProject } from '../../../hooks/api/project.hook';
import { LabelSelection } from '../label-selection/label-selection.component';
import type { TaskType } from '../task-selection/interface';
import { TaskSelection } from '../task-selection/task-selection.component';
import { validateProjectName } from './validator';

import styles from './create-project-form.module.scss';

type CreateProjectFormProps = {
    projects: Project[];
};

export const CreateProjectForm = ({ projects }: CreateProjectFormProps) => {
    const [selectedTask, setSelectedTask] = useState<TaskType | null>(null);
    const [labels, setLabels] = useState<Label[]>([{ id: uuid(), color: '#F20004', name: 'Object' }]);
    const numberOfProjects = projects.length;
    const [name, setName] = useState<string>(`Project #${numberOfProjects + 1}`);

    const navigate = useNavigate();
    const createProjectMutation = useCreateProject();

    const validationErrorMessage = validateProjectName(
        name,
        projects.map((project) => project.name)
    );

    const isCreateProjectDisabled = selectedTask === null || validationErrorMessage !== undefined;

    const createProject = (e: FormEvent) => {
        e.preventDefault();

        if (isCreateProjectDisabled) {
            return;
        }

        const projectId = uuid();

        createProjectMutation.mutate(
            {
                body: {
                    id: projectId,
                    task: {
                        task_type: selectedTask,
                        exclusive_labels: selectedTask === 'classification',
                        labels,
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
        <Form onSubmit={createProject} maxWidth={'80vw'} margin={'0 auto'} validationBehavior={'native'}>
            <Flex
                direction={'column'}
                gap={'size-500'}
                alignItems={'center'}
                marginTop={'size-1000'}
                marginBottom={'size-300'}
            >
                <TextField
                    isRequired
                    value={name}
                    onChange={setName}
                    width={'50%'}
                    errorMessage={validationErrorMessage}
                    validationState={validationErrorMessage === undefined ? undefined : 'invalid'}
                />

                <Text UNSAFE_className={styles.taskTypeSelectionTitle}>
                    What type of task would you like the model to perform?
                </Text>
            </Flex>

            <Flex direction='column' gap='size-300' UNSAFE_style={{ overflow: 'auto', margin: '0 auto' }}>
                <TaskSelection selectedTask={selectedTask} setSelectedTask={setSelectedTask} />

                <LabelSelection labels={labels} setLabels={setLabels} />
            </Flex>

            <Flex justifyContent={'end'} UNSAFE_className={styles.buttonGroup}>
                <ButtonGroup>
                    <Button type={'submit'} variant='accent' isDisabled={isCreateProjectDisabled}>
                        Create project
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};
