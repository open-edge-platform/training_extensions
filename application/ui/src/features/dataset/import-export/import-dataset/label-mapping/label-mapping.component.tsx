// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Grid, Heading, Item, Picker, Text, View } from '@geti/ui';
import { useLabelMappingImportDataset } from 'hooks/localStorage/use-label-mapping-import-dataset.hook';

import { $api } from '../../../../../api/client';
import { DatasetStatistics } from '../../../../../components/dataset-statistics/dataset-statistics.component';
import { useProject } from '../../../../../hooks/api/project.hook';
import { isNonEmptyString } from '../../../../../shared/util';
import { ImportDatasetState } from '../util';

type LabelMappingProps = {
    onNextStep: (step: ImportDatasetState) => void;
};

export const LabelMapping = ({ onNextStep: _onNextStep }: LabelMappingProps) => {
    const { data: selectedProject } = useProject();
    const { getLsLabelMappingImport } = useLabelMappingImportDataset();

    const { stagedDatasetId } = getLsLabelMappingImport() ?? {};

    const { data } = $api.useQuery('get', '/api/staged_datasets/{staged_dataset_id}', {
        params: { path: { staged_dataset_id: String(stagedDatasetId) } },
        enabled: isNonEmptyString(stagedDatasetId),
    });

    const datasetLabels = data?.metadata?.labels ?? [];
    const totalDatasetItems = data?.metadata?.num_items ?? 0;
    const totalAnnotatedItems = data?.metadata?.num_annotations ?? 0;
    const projectLabels = selectedProject.task.labels ?? [];

    return (
        <Flex direction={'column'} gap={'size-200'} UNSAFE_style={{ padding: dimensionValue('size-275') }}>
            <Heading>Imported dataset statistics</Heading>

            <View padding={'size-200'} borderRadius={'regular'} backgroundColor={'gray-75'}>
                <DatasetStatistics totalMediaItems={totalDatasetItems} totalAnnotatedItems={totalAnnotatedItems} />
            </View>

            <Heading marginTop={'size-200'}>Label mapping</Heading>
            <View backgroundColor={'gray-75'} padding={'size-200'} borderRadius={'regular'}>
                <Grid
                    gap={'size-150'}
                    width={'100%'}
                    alignItems={'center'}
                    columns={[`1fr ${dimensionValue('size-400')} 1fr`]}
                >
                    <View>Existing labels</View>
                    <View />
                    <View>Target labels</View>

                    {datasetLabels.map((label) => (
                        <>
                            <Text>{label}</Text>
                            <View>→</View>
                            <View>
                                <Picker name='targetLabel' items={projectLabels} aria-label='project labels list'>
                                    {(item) => <Item key={item.id}>{item.name}</Item>}
                                </Picker>
                            </View>
                        </>
                    ))}
                </Grid>
            </View>
        </Flex>
    );
};
