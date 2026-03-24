// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FormEvent, useState } from 'react';

import { Button, ButtonGroup, Divider, Flex, Form, Text, TextField, toast } from '@geti/ui';
import { Link, useNavigate } from 'react-router-dom';
import { v4 as uuid } from 'uuid';

import { paths } from '../../../constants/paths';
import type { Label, Project, TaskType } from '../../../constants/shared-types';
import { useCreateProject } from '../../../hooks/api/project.hook';
import { LabelSelection } from '../label-selection/label-selection.component';
import { TASK_OPTIONS, TaskSelection } from '../task-selection/task-selection.component';
import { isClassificationTask } from '../task-type-guards';
import {
    ClassificationTaskSelection,
    ClassificationTaskType,
} from './classification-label-selection/classification-task-type-selection.component';
import { validateProjectName } from './validator';

import classes from './create-project-form.module.scss';

type CreateProjectFormProps = {
    projects: Project[];
};

export const CreateProjectForm = ({ projects }: CreateProjectFormProps) => {
    const [selectedTask, setSelectedTask] = useState<TaskType | null>(null);
    const [labels, setLabels] = useState<Label[]>([]);
    const numberOfProjects = projects.length;
    const [name, setName] = useState<string>(`Project #${numberOfProjects + 1}`);
    const selectedTaskOption = TASK_OPTIONS.find((task) => task.value === selectedTask);

    const [classificationTaskType, setClassificationTaskType] = useState<ClassificationTaskType>('single-label');

    const navigate = useNavigate();
    const createProjectMutation = useCreateProject();

    const validationErrorMessage = validateProjectName(
        name,
        projects.map((project) => project.name)
    );

    const isMultiClassProject = isClassificationTask(selectedTask) && classificationTaskType === 'single-label';
    const needsMinimumNumberOfLabels = isMultiClassProject && labels.length < 2;

    const isCreateProjectDisabled =
        selectedTask === null || validationErrorMessage !== undefined || labels.length === 0;

    const createProject = (e: FormEvent) => {
        e.preventDefault();

        if (isCreateProjectDisabled) {
            return;
        }

        if (needsMinimumNumberOfLabels) {
            toast({
                message: 'At least 2 labels are required for single-label classification',
                type: 'warning',
            });

            return;
        }

        const projectId = uuid();

        createProjectMutation.mutate(
            {
                body: {
                    id: projectId,
                    task: {
                        task_type: selectedTask,
                        exclusive_labels:
                            selectedTask === 'classification' && classificationTaskType === 'single-label',
                        labels,
                    },
                    name,
                },
            },
            {
                onSuccess: () => {
                    navigate(paths.project.dataset.index({ projectId }));
                },
            }
        );
    };

    return (
        <Form onSubmit={createProject} validationBehavior={'native'} height={'100%'}>
            <Flex
                flex={1}
                minHeight={0}
                width={'clamp(912px, 60vw, 1052px)'}
                margin={'0 auto'}
                gap={'size-500'}
                direction={'column'}
            >
                <Flex justifyContent={'center'} marginTop={'size-600'}>
                    <TextField
                        aria-label={'Project name input'}
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
                    <Text UNSAFE_className={classes.taskTypeSelectionTitle}>
                        What type of task would you like the model to perform?
                    </Text>

                    <TaskSelection selectedTask={selectedTask} setSelectedTask={setSelectedTask} />

                    {isClassificationTask(selectedTask) && (
                        <ClassificationTaskSelection
                            selectedType={classificationTaskType}
                            onSelectedTypeChange={setClassificationTaskType}
                        />
                    )}

                    {selectedTask !== null && (
                        <Flex direction={'column'} alignItems={'center'} gap={'size-350'}>
                            <Flex>
                                <Text UNSAFE_className={classes.objectsToLearnTitle}>
                                    {`What objects should the model learn to ${selectedTaskOption?.verb}?`}
                                </Text>
                            </Flex>
                            <LabelSelection labels={labels} setLabels={setLabels} taskType={selectedTask} />
                        </Flex>
                    )}
                </Flex>
            </Flex>

            <Flex direction={'column'} alignItems={'center'} UNSAFE_className={classes.buttonGroup} gap={'size-300'}>
                <Divider size={'S'} width={'100%'} />
                <ButtonGroup>
                    <Button variant={'secondary'}>
                        <Link className={classes.link} to={paths.project.index({})}>
                            Go back
                        </Link>
                    </Button>
                    <Button type={'submit'} variant='accent' isDisabled={isCreateProjectDisabled}>
                        Create project
                    </Button>
                </ButtonGroup>
            </Flex>
        </Form>
    );
};
