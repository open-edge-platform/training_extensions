// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment, useActionState } from 'react';

import { dimensionValue, Flex, Form, Grid, Heading, Item, Picker, Text, View } from '@geti/ui';

import { $api } from '../../../../../api/client';
import { DatasetStatistics } from '../../../../../components/dataset-statistics/dataset-statistics.component';
import { useProject } from '../../../../../hooks/api/project.hook';
import { useImportDatasetToProject } from '../../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import { isNonEmptyString } from '../../../../../shared/util';
import { useImportDatasetDialogState } from '../../../providers/export-import-dataset-dialog-provider.component';
import { IMPORT_DATASET_FORM_ID, mapProjectLabels } from './util';

type LabelMappingProps = {
    stagedDatasetId: string;
};

export const LabelMapping = ({ stagedDatasetId }: LabelMappingProps) => {
    const { data: selectedProject } = useProject();
    const { updateImportEntry } = useImportDatasetToProject();
    const { datasetImportDialogState } = useImportDatasetDialogState();

    const { data } = $api.useQuery('get', '/api/staged_datasets/{staged_dataset_id}', {
        params: { path: { staged_dataset_id: stagedDatasetId } },
        enabled: isNonEmptyString(stagedDatasetId),
    });

    const importDatasetJobMutation = $api.useMutation('post', '/api/jobs');

    const datasetLabels = data?.metadata?.labels ?? [];
    const projectLabels = selectedProject?.task?.labels ?? [];
    const totalDatasetItems = data?.metadata?.num_items ?? 0;
    const totalAnnotatedItems = data?.metadata?.num_annotations ?? 0;

    const [_labelsMapping, submitAction] = useActionState<unknown, FormData>(async (_prevState, formData) => {
        await importDatasetJobMutation.mutateAsync(
            {
                body: {
                    job_type: 'import_dataset_to_project',
                    project_id: String(selectedProject?.id),
                    staged_dataset_id: stagedDatasetId,
                    parameters: {
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
    }, {});

    return (
        <Flex direction={'column'} gap={'size-200'} UNSAFE_style={{ padding: dimensionValue('size-275') }}>
            <Heading>Imported dataset statistics</Heading>

            <View padding={'size-200'} borderRadius={'regular'} backgroundColor={'gray-75'}>
                <DatasetStatistics totalMediaItems={totalDatasetItems} totalAnnotatedItems={totalAnnotatedItems} />
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
                        <View>Existing labels</View>
                        <View />
                        <View>Target labels</View>

                        {datasetLabels.map((label, index) => (
                            <Fragment key={label}>
                                <Text>{label}</Text>
                                <View>→</View>
                                <View>
                                    <Picker
                                        name={`targetLabel-${index}`}
                                        items={projectLabels}
                                        aria-label={`Target label for ${label}`}
                                    >
                                        {(item) => <Item key={item.name}>{item.name}</Item>}
                                    </Picker>
                                </View>
                            </Fragment>
                        ))}
                    </Grid>
                </Form>
            </View>
        </Flex>
    );
};
