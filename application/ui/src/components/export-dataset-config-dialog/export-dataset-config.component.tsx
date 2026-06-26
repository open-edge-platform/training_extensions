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
} from '@geti/ui';
import { Alert, LinkOut } from '@geti/ui/icons';
import { OverlayTriggerState } from '@react-stately/overlays';
import { useProject } from 'hooks/api/project.hook';

import { useExportDatasetJobAction } from '../../hooks/use-export-dataset-job-action.hook';
import { Link } from '../../platform/components/link.component';
import { getEmptyLabel } from '../../shared/annotator/labels';
import { MultiSelectList } from '../multi-select-list/multi-select-list.component';
import { getFormatOptions } from '../util';

import classes from './export-dataset-config.module.scss';

type WarningMessagesProps = {
    emptyLabelName: string | null;
};

const WarningMessages = ({ emptyLabelName }: WarningMessagesProps) => {
    return (
        <Flex alignItems={'start'} marginTop={'size-100'} gap={'size-100'}>
            <Flex>
                <Alert className={classes.warningMessageIcon} />
            </Flex>
            <Flex direction={'column'} gap={'size-75'}>
                <Text>
                    Exporting videos is not supported by this dataset format. All annotated frames from videos will be
                    exported as images.
                </Text>
                {emptyLabelName !== null && (
                    <Text>
                        {`Empty labels ('${emptyLabelName}') are exclusively supported by the Geti format. Other
                        export formats do not support empty labels.`}
                    </Text>
                )}
            </Flex>
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

type ExportDatasetDialogContentProps = {
    name: string;
    datasetId: string | null;
    statistics: ReactNode;
    dialogState: OverlayTriggerState;
};

const ExportDatasetDialogContent = ({ name, datasetId, statistics, dialogState }: ExportDatasetDialogContentProps) => {
    const { data: selectedProject } = useProject();

    const [formState, submitAction, isPending] = useExportDatasetJobAction({
        datasetId,
        onSuccess: dialogState.close,
    });

    const formatOptions = getFormatOptions(selectedProject.task.task_type);
    const [selectedExportFormat, setSelectedExportFormat] = useState<string | null>(formatOptions.at(0)?.value ?? null);

    const labels = selectedProject.task.labels?.map((label) => ({ id: label.name, name: label.name })) ?? [];

    const isNonGetiFormatSelected = selectedExportFormat !== 'geti';
    const emptyLabel = getEmptyLabel(selectedProject.task.task_type, selectedProject.task.exclusive_labels);

    return (
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

                    {isNonGetiFormatSelected && <WarningMessages emptyLabelName={emptyLabel?.name ?? null} />}

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
                <Button type='submit' form={FORM_ID} variant='accent' isPending={isPending} isDisabled={isPending}>
                    Export
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

export const ExportDatasetConfig = ({
    name = 'dataset',
    datasetId,
    statistics,
    dialogState,
}: ExportDatasetConfigProps) => {
    return (
        <DialogContainer onDismiss={dialogState.close}>
            {dialogState.isOpen && (
                <ExportDatasetDialogContent
                    name={name}
                    datasetId={datasetId}
                    statistics={statistics}
                    dialogState={dialogState}
                />
            )}
        </DialogContainer>
    );
};
