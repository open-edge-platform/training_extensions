// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useExportDataset } from '../../../../hooks/localStorage/use-export-dataset.hook';
import { ExportJob } from './export-job/export-job.component';

const isMainDataset = <T extends { datasetId: string | null }>({ datasetId }: T) => datasetId === null;

export const ExportJobsList = () => {
    const { getLsExportIds } = useExportDataset();

    const exportItems = getLsExportIds() ?? [];
    const filteredExportItems = exportItems.filter(isMainDataset);

    if (isEmpty(filteredExportItems)) {
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
            {filteredExportItems.toReversed().map((item) => (
                <ExportJob key={item.jobId} jobId={item.jobId} />
            ))}
        </Flex>
    );
};
