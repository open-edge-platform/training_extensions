// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Divider, Flex, Grid, Text } from '@geti-ui/ui';
import { isEmpty, isNil } from 'lodash-es';

import { ExportDatasetMetadata } from '../../../../../../constants/shared-types';
import { useProject } from '../../../../../../hooks/api/project.hook';

type ExportJobDetailsProps = {
    datasetName?: string;
    metadata: ExportDatasetMetadata;
};

const isGetiFormat = (format?: string | null) => format?.toLowerCase() === 'geti';

export const ExportJobDetails = ({ datasetName, metadata }: ExportJobDetailsProps) => {
    const { data: selectedProject } = useProject();

    const projectLabels = selectedProject.task.labels ?? [];
    const exportLabelsNames = metadata.filters.labels ?? [];

    const projectLabelsNames = projectLabels.map((label) => label.name);
    const selectedLabels = exportLabelsNames.filter((name) => projectLabelsNames.includes(name));

    const labelsList = isEmpty(selectedLabels) ? projectLabelsNames : selectedLabels;

    return (
        <Flex direction={'column'}>
            <Text UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-225') }}>
                Export {isNil(datasetName) ? 'dataset' : datasetName}
            </Text>

            <Grid
                marginTop={'size-200'}
                alignItems={'center'}
                gap='size-125'
                columns={['auto', '1px', 'auto', '1px', '1fr']}
            >
                <Text>
                    Format:{' '}
                    <Text
                        UNSAFE_style={{
                            textTransform: isGetiFormat(metadata.export_format) ? 'capitalize' : 'uppercase',
                        }}
                    >
                        {metadata.export_format}
                    </Text>{' '}
                </Text>

                <Divider orientation='vertical' size='S' />

                <Text>Media: {metadata.filters.include_unannotated ? 'All media' : 'Only media with annotations'}</Text>

                <Divider orientation='vertical' size='S' />

                <Text UNSAFE_style={{ overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
                    Labels: {labelsList.join(', ')}
                </Text>
            </Grid>
        </Flex>
    );
};
