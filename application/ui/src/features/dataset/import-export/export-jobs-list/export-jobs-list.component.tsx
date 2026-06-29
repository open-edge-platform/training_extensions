// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti-ui/ui';
import { isEmpty } from 'lodash-es';

import { useExportDataset } from '../../../../hooks/storage/use-export-dataset.hook';
import { ExportJob } from './export-job/export-job.component';

type ExportJobsListProps = {
    predicate: (item: { datasetId: string | null }) => boolean;
};
export const ExportJobsList = ({ predicate }: ExportJobsListProps) => {
    const { getLsExportIds } = useExportDataset();

    const exportItems = getLsExportIds() ?? [];
    const filteredExportItems = exportItems.filter(predicate);

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
                <ExportJob key={item.jobId} jobId={item.jobId} datasetId={item.datasetId} />
            ))}
        </Flex>
    );
};
