// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { useImportDatasetToProject } from 'hooks/localStorage/use-import-dataset-to-project.hook';
import { partition } from 'lodash-es';

import { PrepareImportDataset } from './prepare-import-dataset.component';
import { StagedImportDataset } from './staged-import-dataset/staged-import-dataset.component';

export const ImportJobsList = () => {
    const { getAllImportEntries } = useImportDatasetToProject();
    const importEntries = getAllImportEntries();

    const [preparingImports, otherItems] = partition(importEntries, ({ step }) => step === 'preparing');
    const stagedImports = otherItems.filter(({ step }) => step === 'labelMapping');

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
            {preparingImportsQueue.map(({ stagedDatasetId }) => (
                <PrepareImportDataset key={`prepare-${stagedDatasetId}`} stagedDatasetId={String(stagedDatasetId)} />
            ))}

            {stagedImportsQueue.map(({ fileName, stagedDatasetId }) => (
                <StagedImportDataset
                    key={`staged-${stagedDatasetId}`}
                    fileName={String(fileName)}
                    stagedDatasetId={String(stagedDatasetId)}
                />
            ))}
        </Flex>
    );
};
