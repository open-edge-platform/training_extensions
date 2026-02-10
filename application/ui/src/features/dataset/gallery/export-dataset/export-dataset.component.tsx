// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    Button,
    ButtonGroup,
    Checkbox,
    CheckboxGroup,
    Content,
    Dialog,
    DialogContainer,
    Form,
    Heading,
} from '@geti/ui';
import { useOverlayTriggerState } from '@react-stately/overlays';
import { useProject } from 'hooks/api/project.hook';

import { components } from '../../../../api/openapi-spec';
import { useExportDatasetJobAction } from './use-export-dataset-job-action.hook';

type DatasetItemSubset = components['schemas']['DatasetItemSubset'];

const SUBSET: DatasetItemSubset[] = ['testing', 'training', 'unassigned', 'validation'];

export const ExportDataset = () => {
    const dialogState = useOverlayTriggerState({});
    const { data: selectedProject } = useProject();
    const [formState, submitAction, isPending] = useExportDatasetJobAction({
        onSuccess: dialogState.close,
    });

    const labels = selectedProject?.task?.labels ?? [];

    return (
        <>
            <Button variant='secondary' onPress={dialogState.open}>
                Export dataset
            </Button>

            <DialogContainer onDismiss={dialogState.close}>
                {dialogState.isOpen && (
                    <Dialog size='S'>
                        <Heading>Export dataset</Heading>
                        <Content>
                            <Form validationBehavior={'native'} action={submitAction}>
                                <Checkbox name='include_unannotated' defaultSelected={formState.include_unannotated}>
                                    Include unannotated
                                </Checkbox>

                                <CheckboxGroup label='labels' name='labels' defaultValue={formState.labels}>
                                    {labels.map((label) => (
                                        <Checkbox key={label.id} value={label.id}>
                                            {label.name}
                                        </Checkbox>
                                    ))}
                                </CheckboxGroup>

                                <CheckboxGroup label='subsets' name='subsets' defaultValue={formState.subsets}>
                                    {SUBSET.map((subset) => (
                                        <Checkbox key={subset} value={subset}>
                                            {subset}
                                        </Checkbox>
                                    ))}
                                </CheckboxGroup>

                                <ButtonGroup>
                                    <Button onPress={close} variant='secondary'>
                                        Cancel
                                    </Button>
                                    <Button type='submit' variant='accent' isPending={isPending} isDisabled={isPending}>
                                        Export
                                    </Button>
                                </ButtonGroup>
                            </Form>
                        </Content>
                    </Dialog>
                )}
            </DialogContainer>
        </>
    );
};
