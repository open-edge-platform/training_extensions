// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Divider, Flex, Form, Text, TextField } from '@geti/ui';
import { useNavigate } from 'react-router';
import { v4 as uuid } from 'uuid';

import { paths } from '../../../constants/paths';
import type { Label, Project } from '../../../constants/shared-types';
import { useCreateProject } from '../../../hooks/api/project.hook';
import { LabelSelection } from '../label-selection/label-selection.component';
import type { TaskType } from '../task-selection/interface';
import { TASK_OPTIONS, TaskSelection } from '../task-selection/task-selection.component';
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
    const selectedTaskOption = TASK_OPTIONS.find((task) => task.value === selectedTask);

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
        <Form
            onSubmit={createProject}
            validationBehavior={'native'}
            UNSAFE_className={styles.formContent}
            height={'100%'}
        >
            <Flex
                flex={1}
                minHeight={0}
                width={'clamp(800px, 60vw, 1052px)'}
                margin={'0 auto'}
                gap={'size-500'}
                direction={'column'}
            >
                <Flex justifyContent={'center'} marginTop={'size-600'}>
                    <TextField
                        isRequired
                        value={name}
                        onChange={setName}
                        width={'50%'}
                        errorMessage={validationErrorMessage}
                        validationState={validationErrorMessage === undefined ? undefined : 'invalid'}
                    />
                </Flex>

                <Flex
                    direction='column'
                    gap='size-300'
                    UNSAFE_style={{ overflow: 'auto', margin: '0 auto' }}
                    width={'100%'}
                >
                    <Text UNSAFE_className={styles.taskTypeSelectionTitle}>
                        What type of task would you like the model to perform?
                    </Text>

                    <TaskSelection selectedTask={selectedTask} setSelectedTask={setSelectedTask} />

                    {selectedTask !== null && (
                        <Flex direction={'column'} alignItems={'center'} gap={'size-350'}>
                            <Flex>
                                <Text UNSAFE_className={styles.objectsToLearnTitle}>
                                    {`What objects should the model learn to ${selectedTaskOption?.verb}?`}
                                </Text>
                            </Flex>
                            <LabelSelection labels={labels} setLabels={setLabels} />
                        </Flex>
                    )}
                </Flex>
            </Flex>

            <Flex direction={'column'} alignItems={'center'} UNSAFE_className={styles.buttonGroup} gap={'size-300'}>
                <Divider size={'S'} width={'100%'} />
                <ButtonGroup>
                    <Button type={'submit'} variant='accent' isDisabled={isCreateProjectDisabled}>
                        Create project
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};
