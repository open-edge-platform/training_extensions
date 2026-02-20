// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useExportDataset } from '../../../../hooks/localStorage/use-export-dataset.hook';
import { ExportJob } from './export-job/export-job.component';

export const ExportJobsList = () => {
    const { getLsExportIds } = useExportDataset();

    const exportIds = getLsExportIds() ?? [];

    if (isEmpty(exportIds)) {
        return null;
    }

    return (
        <Flex
            gap='size-250'
            direction='column'
            maxHeight='size-3400'
            marginBottom='size-250'
            UNSAFE_style={{ overflowY: 'auto' }}
        >
            {exportIds.toReversed().map((id) => (
                <ExportJob key={id} jobId={id} />
            ))}
        </Flex>
    );
};
