// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { useImportDatasetAsNewProject } from 'hooks/localStorage/use-import-dataset-as-new-project.hook';
import { partition } from 'lodash-es';

import { PrepareImportDataset } from '../../../../components/prepare-import-dataset/prepare-import-dataset.component';
import { StagedImportDataset } from './staged-import-dataset/staged-import-dataset.component';

export const ImportJobsList = () => {
    const { getAllImportEntries, deleteImportEntry, updateImportEntryStep } = useImportDatasetAsNewProject();

    const importEntries = getAllImportEntries();

    const [preparingImports, others] = partition(importEntries, ({ step }) => step === 'preparing');
    const [taskTypeSelectionImports, restItems] = partition(others, ({ step }) => step === 'taskTypeSelection');
    const [labelMappingImports] = partition(restItems, ({ step }) => step === 'labelMapping');

    const preparingImportsQueue = preparingImports.reverse();
    const labelMappingImportsQueue = labelMappingImports.reverse();
    const taskTypeSelectionImportsQueue = taskTypeSelectionImports.reverse();

    return (
        <Flex
            gap='size-250'
            maxHeight='228px'
            direction='column'
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

            {taskTypeSelectionImportsQueue.map(({ fileName, stagedDatasetId }) => {
                return (
                    <StagedImportDataset
                        key={`task-type-${stagedDatasetId}`}
                        fileName={fileName}
                        message={'Select task type'}
                        openState={'taskTypeSelection'}
                        stagedDatasetId={stagedDatasetId}
                        primaryButtonLabel={'Select task type'}
                    />
                );
            })}

            {labelMappingImportsQueue.map(({ fileName, stagedDatasetId }) => {
                return (
                    <StagedImportDataset
                        key={`label-mapping-${stagedDatasetId}`}
                        fileName={fileName}
                        message={'Map labels for the uploaded dataset'}
                        openState={'labelMapping'}
                        stagedDatasetId={stagedDatasetId}
                        primaryButtonLabel={'Map labels'}
                    />
                );
            })}
        </Flex>
    );
};
