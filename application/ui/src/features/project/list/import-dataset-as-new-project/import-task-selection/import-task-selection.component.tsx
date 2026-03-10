// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { Flex, Form, Item, Picker, Text, TextField, View } from '@geti/ui';
import { InfoOutline } from '@geti/ui/icons';
import { useStagedDataset } from 'hooks/api/staged-dataset.hook';

import { useImportDatasetDialog } from '../../../providers/import-dataset-dialog-provider.component';
import { getRecommendedTaskType, TASK_SELECTION_FORM_ID } from './util';

type ImportTaskSelectionProps = {
    stagedDatasetId: string;
};

export const ImportTaskSelection = ({ stagedDatasetId }: ImportTaskSelectionProps) => {
    const { setCurrentStep } = useImportDatasetDialog();
    const { data: stagedDataset } = useStagedDataset(stagedDatasetId);

    const defaultTaskType = getRecommendedTaskType(stagedDataset?.metadata?.annotation_type);

    const [formState, submitAction] = useActionState<{ name: string; task_type: string }, FormData>(
        async (_prevState, formData) => {
            const data = {
                name: String(formData.get('name')),
                task_type: String(formData.get('task_type')),
            };
            console.log('data', data);

            setCurrentStep('labelMapping');
            return data;
        },
        { name: 'Project #1', task_type: defaultTaskType || '' }
    );

    return (
        <View backgroundColor={'gray-75'} margin={'size-300'} padding={'size-300'}>
            <Form id={TASK_SELECTION_FORM_ID} validationBehavior='native' action={submitAction}>
                <TextField
                    isRequired
                    name={'name'}
                    label={'Project name'}
                    defaultValue={formState.name}
                    marginBottom={'size-250'}
                />

                <Picker
                    isRequired
                    name={'task_type'}
                    label={'Task type'}
                    aria-label={'Task type'}
                    marginBottom={'size-150'}
                    defaultSelectedKey={defaultTaskType}
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

                <Flex gap='size-100'>
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
