// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useImportDatasetAsNewProject } from 'hooks/localStorage/use-import-dataset-as-new-project.hook';
import { partition } from 'lodash-es';

import { LoadingImportDataset } from '../../../../components/loading-import-dataset/loading-import-dataset.component';
import { PrepareImportDataset } from '../../../../components/prepare-import-dataset/prepare-import-dataset.component';
import { getQueryKey } from '../../../../query-client/query-client';
import { StagedImportDataset } from './staged-import-dataset/staged-import-dataset.component';

export const ImportJobsList = () => {
    const queryClient = useQueryClient();
    const { getAllImportEntries, deleteImportEntry, updateImportEntryStep } = useImportDatasetAsNewProject();

    const importEntries = getAllImportEntries();

    const [preparingImports, others] = partition(importEntries, ({ step }) => step === 'preparing');
    const [taskTypeSelectionImports, restItems] = partition(others, ({ step }) => step === 'taskTypeSelection');
    const [labelMappingImports, leftOvers] = partition(restItems, ({ step }) => step === 'labelMapping');
    const [importingJob] = partition(leftOvers, ({ step }) => step === 'importing');

    const importingJobQueue = importingJob.reverse();
    const preparingImportsQueue = preparingImports.reverse();
    const labelMappingImportsQueue = labelMappingImports.reverse();
    const taskTypeSelectionImportsQueue = taskTypeSelectionImports.reverse();

    const handleImportSuccess = () => {
        queryClient.invalidateQueries({
            queryKey: getQueryKey(['get', '/api/projects']),
        });
    };

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

            {importingJobQueue.map(({ size, fileName, stagedDatasetId, importJobId }) => (
                <LoadingImportDataset
                    key={`loading-${stagedDatasetId}`}
                    size={size}
                    fileName={fileName}
                    jobId={String(importJobId)}
                    stagedDatasetId={String(stagedDatasetId)}
                    onSuccess={handleImportSuccess}
                    deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                />
            ))}
        </Flex>
    );
};
