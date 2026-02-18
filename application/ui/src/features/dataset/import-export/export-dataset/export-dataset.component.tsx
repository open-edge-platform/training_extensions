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
    Radio,
    RadioGroup,
    Link as SpectrumLink,
    View,
} from '@geti/ui';
import { LinkOut } from '@geti/ui/icons';
import { OverlayTriggerState } from '@react-stately/overlays';
import { useProject } from 'hooks/api/project.hook';

import { MultiSelectList } from '../../../../components/multi-select-list/multi-select-list.component';
import { isClassificationTask } from '../../../project/task-type-guards';
import { DatasetStatistics } from './dataset-statistics/dataset-statistics.component';
import { useExportDatasetJobAction } from './hooks/use-export-dataset-job-action.hook';

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

    const labels = selectedProject.task?.labels?.map((label) => ({ id: label.name, name: label.name })) ?? [];

    const typeItems = [
        { label: 'GETI', value: 'geti' },
        { label: 'YOLO', value: 'yolo' },
        { label: 'COCO', value: 'coco' },
    ].filter((item) => item.value !== 'coco' || !isClassificationTask(selectedProject.task.task_type));

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
                            <Form id={FORM_ID} validationBehavior='native' action={submitAction}>
                                <MultiSelectList
                                    name='labels'
                                    items={labels}
                                    maxHeight='size-2000'
                                    label='Choose images by label to export'
                                />

                                <Checkbox name='include_unannotated' defaultSelected={formState.include_unannotated}>
                                    Include unannotated
                                </Checkbox>

                                <Divider size='S' />

                                <RadioGroup
                                    name='export_format'
                                    label='Select dataset export format'
                                    defaultValue={formState.export_format}
                                >
                                    {typeItems.map((item) => (
                                        <Radio key={item.value} value={item.value}>
                                            {item.label}
                                        </Radio>
                                    ))}
                                </RadioGroup>
                            </Form>

                            {/* TODO: pending link url
                            https://github.com/open-edge-platform/training_extensions/issues/5512 */}
                            <SpectrumLink UNSAFE_className={classes.link}>
                                <a href={'/'} target={'_blank'} rel={'noopener noreferrer'}>
                                    Learn more about export formats
                                    <LinkOut size='XS' />
                                </a>
                            </SpectrumLink>
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
