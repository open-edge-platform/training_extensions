// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { useImportDatasetAsNewProject } from 'hooks/localStorage/use-import-dataset-as-new-project.hook';
import { partition } from 'lodash-es';

import { PrepareImportDataset } from '../../../../components/prepare-import-dataset/prepare-import-dataset.component';

export const ImportJobsList = () => {
    const { getAllImportEntries, deleteImportEntry, updateImportEntryStep } = useImportDatasetAsNewProject();

    const importEntries = getAllImportEntries();

    const [preparingImports] = partition(importEntries, ({ step }) => step === 'preparing');

    const preparingImportsQueue = preparingImports.reverse();

    return (
        <Flex
            gap='size-250'
            direction='column'
            maxHeight='size-3400'
            marginBottom='size-250'
            UNSAFE_style={{ overflowY: 'auto' }}
        >
            {preparingImportsQueue.map(({ size, fileName, stagedDatasetId, prepareJobId }) => {
                return (
                    <PrepareImportDataset
                        key={`prepare-${stagedDatasetId}`}
                        size={size}
                        fileName={fileName}
                        jobId={prepareJobId}
                        stagedDatasetId={stagedDatasetId}
                        onSuccess={() => updateImportEntryStep(stagedDatasetId, 'taskTypeSelection')}
                        deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                    />
                );
            })}
        </Flex>
    );
};
