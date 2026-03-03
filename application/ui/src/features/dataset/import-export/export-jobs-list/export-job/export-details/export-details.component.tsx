// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Divider, Grid, Text } from '@geti/ui';
import { isEmpty, isNil } from 'lodash-es';

import { ExportDatasetMetadata } from '../../../../../../constants/shared-types';
import { useProject } from '../../../../../../hooks/api/project.hook';

type ExportJobDetailsProps = {
    datasetName?: string;
    metadata: ExportDatasetMetadata;
};
export const ExportJobDetails = ({ datasetName, metadata }: ExportJobDetailsProps) => {
    const { data: selectedProject } = useProject();

    const projectLabels = selectedProject.task.labels ?? [];
    const exportLabelsNames = metadata.filters.labels ?? [];

    const projectLabelsNames = projectLabels.map((label) => label.name);
    const selectedLabels = exportLabelsNames.filter((name) => projectLabelsNames.includes(name));

    const labelsList = isEmpty(selectedLabels) ? projectLabelsNames : selectedLabels;

    return (
        <Grid
            gap='size-125'
            columns={['auto', '1px', '1fr', '1px', 'auto']}
            UNSAFE_style={{ fontWeight: 500, fontSize: dimensionValue('size-200') }}
        >
            <Text>
                Export {isNil(datasetName) ? 'dataset' : datasetName} - {metadata.export_format} format
            </Text>

            <Divider orientation='vertical' size='S' />
            <Text UNSAFE_style={{ overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
                {`Included images by label: ${labelsList.join(', ')}`}
            </Text>

            <Divider orientation='vertical' size='S' />
            <Text>{metadata.filters.include_unannotated ? 'Include:' : 'Exclude:'} media without annotations</Text>
        </Grid>
    );
};
