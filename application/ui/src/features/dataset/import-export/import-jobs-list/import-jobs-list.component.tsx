// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { useImportDatasetToProject } from 'hooks/localStorage/use-import-dataset-to-project.hook';
import { partition } from 'lodash-es';

import { isNonEmptyString } from '../../../../shared/util';
import { PrepareImportDataset } from './prepare-import-dataset.component';
import { StagedImportDataset } from './staged-import-dataset/staged-import-dataset.component';

export const ImportJobsList = () => {
    const { getAllImportEntries } = useImportDatasetToProject();
    const importEntries = getAllImportEntries();

    const [preparingImports, otherItems] = partition(importEntries, (item) => isNonEmptyString(item.prepareJobId));
    const stagedImports = otherItems.filter((item) => isNonEmptyString(item.stagedDatasetId));

    return (
        <Flex
            gap='size-250'
            direction='column'
            maxHeight='size-3400'
            marginBottom='size-250'
            UNSAFE_style={{ overflowY: 'auto' }}
        >
            {preparingImports.map(({ prepareJobId }) => (
                <PrepareImportDataset key={prepareJobId} prepareJobId={String(prepareJobId)} />
            ))}

            {stagedImports.map(({ fileName, stagedDatasetId }) => (
                <StagedImportDataset
                    key={stagedDatasetId}
                    fileName={String(fileName)}
                    stagedDatasetId={String(stagedDatasetId)}
                />
            ))}
        </Flex>
    );
};
