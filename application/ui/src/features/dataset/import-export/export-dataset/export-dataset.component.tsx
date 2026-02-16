// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    Button,
    ButtonGroup,
    Checkbox,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Form,
    Heading,
    Item,
    Picker,
    Radio,
    RadioGroup,
    View,
} from '@geti/ui';
import { OverlayTriggerState } from '@react-stately/overlays';
import { useProject } from 'hooks/api/project.hook';

import { DatasetStatistics } from './dataset-statistics/dataset-statistics.component';
import { useExportDatasetJobAction } from './use-export-dataset-job-action.hook';

import classes from './export-dataset.module.scss';

type ExportDatasetProps = {
    dialogState: OverlayTriggerState;
};

const FORM_ID = 'export-dataset-form';

export const ExportDataset = ({ dialogState }: ExportDatasetProps) => {
    const { data: selectedProject } = useProject();
    const [formState, submitAction, isPending] = useExportDatasetJobAction({
        onSuccess: dialogState.close,
    });

    const labels = selectedProject.task?.labels ?? [];

    return (
        <DialogContainer onDismiss={dialogState.close}>
            {dialogState.isOpen && (
                <Dialog size='L'>
                    <Heading>Export dataset</Heading>
                    <Divider />
                    <Content UNSAFE_className={classes.container}>
                        <Heading>Exported dataset statistics</Heading>
                        <DatasetStatistics />

                        <Heading>Export settings</Heading>

                        <View backgroundColor='gray-75' padding='size-200' borderRadius='regular'>
                            <Form id={FORM_ID} validationBehavior={'native'} action={submitAction}>
                                {/* TODO: temporary, the final implementation will need a custom implementation */}
                                <Picker
                                    name='labels'
                                    label='Choose images by label to export'
                                    defaultSelectedKey={formState.labels[0]}
                                >
                                    {labels.map((label) => (
                                        <Item key={label.id}>{label.name}</Item>
                                    ))}
                                </Picker>

                                <Checkbox name='include_unannotated' defaultSelected={formState.include_unannotated}>
                                    Include unannotated
                                </Checkbox>

                                <Divider size='S' />

                                <RadioGroup
                                    name='export_format'
                                    label='Select dataset export format'
                                    defaultValue={formState.export_format}
                                >
                                    <Radio value='geti'>GETI</Radio>
                                    <Radio value='yolo'>YOLO</Radio>
                                    <Radio value='coco'>COCO</Radio>
                                </RadioGroup>
                            </Form>
                        </View>
                    </Content>

                    <ButtonGroup>
                        <Button onPress={dialogState.close} variant='secondary'>
                            Cancel
                        </Button>
                        <Button
                            type='submit'
                            form={FORM_ID}
                            variant='accent'
                            isPending={isPending}
                            isDisabled={isPending}
                        >
                            Export
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogContainer>
    );
};
