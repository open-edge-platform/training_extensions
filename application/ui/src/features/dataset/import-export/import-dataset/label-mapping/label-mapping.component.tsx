// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment, useActionState } from 'react';

import { Checkbox, dimensionValue, Flex, Form, Grid, Heading, Item, Picker, Text, View } from '@geti/ui';
import { useStagedDataset } from 'hooks/api/staged-dataset.hook';

import { $api } from '../../../../../api/client';
import { DatasetStatistics } from '../../../../../components/dataset-statistics/dataset-statistics.component';
import { useProject } from '../../../../../hooks/api/project.hook';
import { useImportDatasetToProject } from '../../../../../hooks/storage/use-import-dataset-to-project.hook';
import { useImportDatasetDialogState } from '../../../providers/export-import-dataset-dialog-provider.component';
import { FormatWarning } from './format-warning/format-warning.component';
import { IMPORT_DATASET_FORM_ID, mapProjectLabels } from './util';

type LabelMappingProps = {
    stagedDatasetId: string;
};

type useFormConfigProps = {
    datasetLabels: string[];
    stagedDatasetId: string;
    selectedProjectId: string;
};

const useFormConfig = ({ datasetLabels, stagedDatasetId, selectedProjectId }: useFormConfigProps) => {
    const { updateImportEntry } = useImportDatasetToProject();
    const { datasetImportDialogState } = useImportDatasetDialogState();
    const importDatasetJobMutation = $api.useMutation('post', '/api/jobs');

    return useActionState<{ include_unannotated: boolean }, FormData>(
        async (_prevState, formData) => {
            const state = { include_unannotated: formData.get('include_unannotated') === 'on' };

            await importDatasetJobMutation.mutateAsync(
                {
                    body: {
                        job_type: 'import_dataset_to_project',
                        project_id: selectedProjectId,
                        staged_dataset_id: stagedDatasetId,
                        parameters: {
                            include_unannotated: state.include_unannotated,
                            labels_mapping: mapProjectLabels(datasetLabels, formData),
                        },
                    },
                },
                {
                    onSuccess: ({ job_id }) => {
                        updateImportEntry(stagedDatasetId, { importJobId: job_id, step: 'importing' });
                    },
                    onSettled: () => {
                        datasetImportDialogState.close();
                    },
                }
            );
            return state;
        },
        { include_unannotated: true }
    );
};

export const LabelMapping = ({ stagedDatasetId }: LabelMappingProps) => {
    const { data: selectedProject } = useProject();
    const projectLabels = selectedProject?.task?.labels ?? [];

    const { data: stagedDataset } = useStagedDataset(stagedDatasetId);

    const datasetLabels = stagedDataset?.metadata?.labels ?? [];
    const totalImages = stagedDataset?.metadata?.num_images ?? 0;
    const totalAnnotatedImages = stagedDataset?.metadata?.num_annotated_images ?? 0;

    const totalFrames = stagedDataset?.metadata?.num_frames ?? 0;
    const totalAnnotatedFrames = stagedDataset?.metadata?.num_annotated_frames ?? 0;

    const [formState, submitAction] = useFormConfig({
        datasetLabels,
        stagedDatasetId,
        selectedProjectId: selectedProject?.id ?? '',
    });

    return (
        <Flex direction={'column'} gap={'size-200'} UNSAFE_style={{ padding: dimensionValue('size-275') }}>
            <Heading>Imported dataset statistics</Heading>

            <View padding={'size-200'} borderRadius={'regular'} backgroundColor={'gray-75'}>
                <Flex justifyContent={'center'} gap={'size-200'}>
                    <DatasetStatistics
                        label='images'
                        totalMediaItems={totalImages}
                        totalAnnotatedItems={totalAnnotatedImages}
                    />

                    {totalFrames > 0 && (
                        <DatasetStatistics
                            label='frames'
                            totalMediaItems={totalFrames}
                            totalAnnotatedItems={totalAnnotatedFrames}
                        />
                    )}
                </Flex>

                <FormatWarning annotationType={stagedDataset?.metadata?.annotation_type} />
            </View>

            <Heading marginTop={'size-200'}>Label mapping</Heading>
            <View backgroundColor={'gray-75'} padding={'size-200'} borderRadius={'regular'}>
                <Form id={IMPORT_DATASET_FORM_ID} validationBehavior='native' action={submitAction}>
                    <Grid
                        gap={'size-150'}
                        width={'100%'}
                        alignItems={'center'}
                        columns={[`1fr ${dimensionValue('size-400')} 1fr`]}
                    >
                        <View>Dataset labels</View>
                        <View />
                        <View>Project labels</View>

                        {datasetLabels.map((label, index) => (
                            <Fragment key={label}>
                                <Text>{label}</Text>
                                <View>→</View>
                                <View>
                                    <Picker
                                        items={projectLabels}
                                        name={`targetLabel-${index}`}
                                        aria-label={`Target label for ${label}`}
                                        defaultSelectedKey={projectLabels.find(({ name }) => name === label)?.name}
                                    >
                                        {(item) => <Item key={item.name}>{item.name}</Item>}
                                    </Picker>
                                </View>
                            </Fragment>
                        ))}
                    </Grid>

                    <Checkbox
                        defaultSelected={formState.include_unannotated}
                        name='include_unannotated'
                        aria-label='include unannotated'
                    >
                        Include media without annotations
                    </Checkbox>
                </Form>
            </View>
        </Flex>
    );
};
