// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState, useState } from 'react';

import { Flex, Form, Item, Picker, Text, TextField, View } from '@geti/ui';
import { InfoOutline } from '@geti/ui/icons';
import { useProjects } from 'hooks/api/project.hook';
import { useStagedDatasetSuspense } from 'hooks/api/staged-dataset.hook';
import { useImportDatasetAsNewProject } from 'hooks/localStorage/use-import-dataset-as-new-project.hook';

import { TaskType } from '../../../../../constants/shared-types';
import { validateProjectName } from '../../../create/validator';
import { useImportDatasetDialog } from '../../../providers/import-dataset-dialog-provider.component';
import { getRecommendedTaskType, TASK_SELECTION_FORM_ID } from './util';

type ImportTaskSelectionProps = {
    stagedDatasetId: string;
};

const useFormConfig = (stagedDatasetId: string, defaultTaskType: TaskType | undefined) => {
    const { data: projects } = useProjects();
    const { setCurrentStep } = useImportDatasetDialog();
    const { getImportEntry, updateImportEntry } = useImportDatasetAsNewProject();
    const importEntry = getImportEntry(stagedDatasetId);

    const initialFormState = {
        name: importEntry?.project?.name ?? `Project #${projects.length + 1}`,
        task_type: importEntry?.project?.task_type ?? defaultTaskType,
    };

    return useActionState<{ name: string; task_type: TaskType | undefined }, FormData>(async (_prevState, formData) => {
        const project = {
            name: String(formData.get('name')).trim(),
            task_type: formData.get('task_type') as TaskType,
        };

        setCurrentStep('labelMapping');
        updateImportEntry(stagedDatasetId, { project, step: 'labelMapping' });
        return project;
    }, initialFormState);
};

export const ImportTaskSelection = ({ stagedDatasetId }: ImportTaskSelectionProps) => {
    const { data: projects } = useProjects();
    const { data: stagedDataset } = useStagedDatasetSuspense(stagedDatasetId);

    const isGetiFormat = stagedDataset.format === 'geti';
    const defaultTaskType = isGetiFormat ? getRecommendedTaskType(stagedDataset?.metadata?.annotation_type) : undefined;

    const [formState, submitAction] = useFormConfig(stagedDatasetId, defaultTaskType);
    const [name, setName] = useState(formState.name);

    const validationErrorMessage = validateProjectName(
        name.trim(),
        projects.map((project) => project.name)
    );

    return (
        <View backgroundColor={'gray-75'} margin={'size-300'} padding={'size-300'}>
            <Form id={TASK_SELECTION_FORM_ID} validationBehavior='native' action={submitAction}>
                <TextField
                    isRequired
                    name={'name'}
                    value={name}
                    onChange={setName}
                    label={'Project name'}
                    aria-label={'Project name'}
                    defaultValue={formState.name}
                    marginBottom={'size-250'}
                    errorMessage={validationErrorMessage}
                    validationState={validationErrorMessage === undefined ? undefined : 'invalid'}
                />

                <Picker
                    isRequired
                    name={'task_type'}
                    label={'Task type'}
                    aria-label={'Task type'}
                    marginBottom={'size-150'}
                    placeholder='select an option...'
                    defaultSelectedKey={formState.task_type}
                >
                    <Item key={'detection'}>
                        {defaultTaskType === 'detection' ? 'Detection (Recommended)' : 'Detection'}
                    </Item>
                    <Item key={'classification'}>
                        {defaultTaskType === 'classification' ? 'Classification (Recommended)' : 'Classification'}
                    </Item>
                    <Item key={'instance_segmentation'}>
                        {defaultTaskType === 'instance_segmentation'
                            ? 'Instance segmentation (Recommended)'
                            : 'Instance segmentation'}
                    </Item>
                </Picker>

                <Flex gap='size-100' alignItems={'center'}>
                    <View width={16} height={16}>
                        <InfoOutline />
                    </View>

                    <Text>
                        The recommended choice is based on the type of the annotations detected in the dataset. If you
                        choose a different type, the annotations will be automatically transformed during import to fit
                        the selected type.
                    </Text>
                </Flex>
            </Form>
        </View>
    );
};
