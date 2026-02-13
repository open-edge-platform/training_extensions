// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Grid, Text } from '@geti/ui';

import { ExportDatasetMetadata } from '../../../../../constants/shared-types';
import { useProject } from '../../../../../hooks/api/project.hook';
import { isNonEmptyArray } from '../../../../../shared/util';

type ExportDetailsProps = {
    metadata: ExportDatasetMetadata;
};
export const ExportDetails = ({ metadata }: ExportDetailsProps) => {
    const { data: selectedProject } = useProject();
    const projectLabels = selectedProject.task.labels ?? [];
    const exportLabels = metadata.filters.labels ?? [];

    const labelsNames = exportLabels
        .map((labelId) => projectLabels.find((label) => label.id === labelId)?.name)
        .filter(Boolean);

    return (
        <Grid gap='size-125' columns={['auto', '1px', '1fr', '1px', 'auto']}>
            <Text>Export dataset - {metadata.export_format} format</Text>

            {isNonEmptyArray(labelsNames) && (
                <>
                    <Divider orientation='vertical' size='S' />
                    <Text UNSAFE_style={{ overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
                        Included images by label: {labelsNames.join(', ')}
                    </Text>
                </>
            )}

            {metadata.filters.include_unannotated === false && (
                <>
                    <Divider orientation='vertical' size='S' />
                    <Text>Excluded: Unannotated</Text>
                </>
            )}
        </Grid>
    );
};
