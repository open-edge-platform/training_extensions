// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { useLocalStorageDataset } from 'hooks/use-local-storage-dataset.hook';
import { isEmpty } from 'lodash-es';

import { ExportJob } from './export-job/export-job.component';

export const ExportJobsList = () => {
    const { getLsExportIds } = useLocalStorageDataset();

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
            {exportIds.map((id) => (
                <ExportJob key={id} jobId={id} />
            ))}
        </Flex>
    );
};
