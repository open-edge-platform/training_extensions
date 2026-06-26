// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import {
    Button,
    ButtonGroup,
    Checkbox,
    Content,
    Dialog,
    DialogContainer,
    Divider,
    Flex,
    Form,
    Heading,
    Radio,
    RadioGroup,
    Text,
    View,
} from '@geti-ui/ui';
import { Alert, LinkOut } from '@geti-ui/ui/icons';
import { OverlayTriggerState } from '@react-stately/overlays';
import { useProject } from 'hooks/api/project.hook';

import { useExportDatasetJobAction } from '../../hooks/use-export-dataset-job-action.hook';
import { Link } from '../../platform/components/link.component';
import { MultiSelectList } from '../multi-select-list/multi-select-list.component';
import { getFormatOptions } from '../util';

import classes from './export-dataset-config.module.scss';

const WarningMessage = () => {
    return (
        <Flex alignItems={'start'} marginTop={'size-100'} gap={'size-100'}>
            <Flex>
                <Alert className={classes.warningMessageIcon} />
            </Flex>
            <Text>
                Exporting videos is not supported by this dataset format. All annotated frames from videos will be
                exported as images.
            </Text>
        </Flex>
    );
};

type ExportDatasetConfigProps = {
    name?: string;
    datasetId: string | null;
    statistics: ReactNode;
    dialogState: OverlayTriggerState;
};

const FORM_ID = 'export-dataset-form';
const EXPORT_FORMATS_LINK =
    'https://docs.geti.intel.com/docs/user-guide/geti-fundamentals/datasets/dataset-export-import#supported-formats';

export const ExportDatasetConfig = ({
    name = 'dataset',
    datasetId,
    statistics,
    dialogState,
}: ExportDatasetConfigProps) => {
    const { data: selectedProject } = useProject();

    const [formState, submitAction, isPending] = useExportDatasetJobAction({
        datasetId,
        onSuccess: dialogState.close,
    });

    const formatOptions = getFormatOptions(selectedProject.task.task_type);
    const [selectedExportFormat, setSelectedExportFormat] = useState<string | null>(formatOptions.at(0)?.value ?? null);

    const labels = selectedProject.task?.labels?.map((label) => ({ id: label.name, name: label.name })) ?? [];

    const isNonGetiFormatSelected = selectedExportFormat !== 'geti';

    return (
        <DialogContainer onDismiss={dialogState.close}>
            {dialogState.isOpen && (
                <Dialog size='L' width={{ base: '70vw' }}>
                    <Heading>Export {name}</Heading>
                    <Divider />
                    <Content UNSAFE_className={classes.container}>
                        <Heading>Exported dataset statistics</Heading>
                        {statistics}

                        <Heading>Export settings</Heading>

                        <View backgroundColor='gray-75' padding='size-200' borderRadius='regular'>
                            <Form id={FORM_ID} validationBehavior='native' action={submitAction}>
                                <MultiSelectList
                                    name='labels'
                                    items={labels}
                                    maxHeight='size-2000'
                                    label='Filter annotations by label'
                                    defaultSelectedKeys={new Set(labels.map(({ id }) => id))}
                                />

                                <Checkbox name='include_unannotated' defaultSelected={formState.include_unannotated}>
                                    Include media without annotations
                                </Checkbox>

                                <Divider size='S' />

                                <RadioGroup
                                    name='export_format'
                                    label='Select dataset export format'
                                    defaultValue={formState.export_format}
                                    onChange={(value) => setSelectedExportFormat(value)}
                                >
                                    {formatOptions.map((item) => (
                                        <Radio key={item.value} value={item.value}>
                                            {item.label}
                                        </Radio>
                                    ))}
                                </RadioGroup>
                            </Form>

                            {isNonGetiFormatSelected && <WarningMessage />}

                            <Link
                                href={EXPORT_FORMATS_LINK}
                                target='_blank'
                                rel='noopener noreferrer'
                                UNSAFE_className={classes.link}
                            >
                                Learn more about export formats
                                <LinkOut size='XS' />
                            </Link>
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
