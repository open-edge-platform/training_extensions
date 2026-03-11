// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { useImportDatasetToProject } from 'hooks/localStorage/use-import-dataset-to-project.hook';
import { partition } from 'lodash-es';

import { PrepareImportDataset } from '../../../../components/prepare-import-dataset/prepare-import-dataset.component';
import { LoadingImportDataset } from './loading-import-dataset/loading-import-dataset.component';
import { StagedImportDataset } from './staged-import-dataset/staged-import-dataset.component';

export const ImportJobsList = () => {
    const { getAllImportEntries, deleteImportEntry, updateImportEntryStep } = useImportDatasetToProject();

    const importEntries = getAllImportEntries();

    const [preparingImports, otherItems] = partition(importEntries, ({ step }) => step === 'preparing');
    const [stagedImports, loadingItems] = partition(otherItems, ({ step }) => step === 'labelMapping');

    const loadingItemsQueue = loadingItems.reverse();
    const stagedImportsQueue = stagedImports.reverse();
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
                        onSuccess={() => updateImportEntryStep(stagedDatasetId, 'labelMapping')}
                        deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                    />
                );
            })}

            {stagedImportsQueue.map(({ fileName, stagedDatasetId }) => (
                <StagedImportDataset
                    key={`staged-${stagedDatasetId}`}
                    fileName={String(fileName)}
                    stagedDatasetId={String(stagedDatasetId)}
                />
            ))}

            {loadingItemsQueue.map(({ stagedDatasetId }) => (
                <LoadingImportDataset key={`loading-${stagedDatasetId}`} stagedDatasetId={String(stagedDatasetId)} />
            ))}
        </Flex>
    );
};
