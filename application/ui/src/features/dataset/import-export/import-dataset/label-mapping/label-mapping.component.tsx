// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Fragment, useActionState } from 'react';

import { Checkbox, dimensionValue, Flex, Form, Grid, Heading, Item, Picker, Text, View } from '@geti/ui';

import { $api } from '../../../../../api/client';
import { DatasetStatistics } from '../../../../../components/dataset-statistics/dataset-statistics.component';
import { useProject } from '../../../../../hooks/api/project.hook';
import { isNonEmptyString } from '../../../../../shared/util';
import { IMPORT_DATASET_FORM_ID } from './util';

type LabelMappingProps = {
    stagedDatasetId: string;
};

type LabelsMapping = Record<string, string>;

export const LabelMapping = ({ stagedDatasetId }: LabelMappingProps) => {
    const { data: selectedProject } = useProject();

    const { data } = $api.useQuery('get', '/api/staged_datasets/{staged_dataset_id}', {
        params: { path: { staged_dataset_id: String(stagedDatasetId) } },
        enabled: isNonEmptyString(stagedDatasetId),
    });

    const datasetLabels = data?.metadata?.labels ?? [];
    const projectLabels = selectedProject?.task?.labels ?? [];
    const totalDatasetItems = data?.metadata?.num_items ?? 0;
    /* 
        Todo: update with totalAnnotatedImages 
        https://github.com/open-edge-platform/training_extensions/issues/5595#issuecomment-3958446137 
    */
    const totalAnnotatedItems = data?.metadata?.num_annotations ?? 0;

    const [_labelsMapping, submitAction] = useActionState<unknown, FormData>(async (_prevState, formData) => {
        const mapping = datasetLabels.reduce<LabelsMapping>((acc, sourceLabel, index) => {
            const targetLabel = formData.get(`targetLabel-${index}`);

            if (isNonEmptyString(targetLabel)) {
                acc[sourceLabel] = targetLabel;
            }
            return acc;
        }, {});
        /* 
         Todo: to implement once the backend support "import_dataset_to_project" jobs

         const response = await importDatasetJobMutation.mutateAsync({
            body: {
                job_type: 'import_dataset_to_project',
                project_id: String(selectedProject?.id),
                staged_dataset_id: String(stagedDatasetId),
                parameters: {
                    filters: { include_unannotated: formData.get('include_unannotated') === 'on' },
                    labels_mapping: mapping,
                },
            },
        }); */

        return mapping;
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

                    <Checkbox name='include_unannotated'>Include media without annotations</Checkbox>
                </Form>
            </View>
        </Flex>
    );
};
