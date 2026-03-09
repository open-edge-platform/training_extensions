// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useImportDatasetAsNewProject } from 'hooks/localStorage/use-import-dataset-as-new-project.hook';

import { ImportJobProcess } from '../../../../../components/import-job-process/import-job-process.component';

type ImportProcessProps = {
    stagedDatasetId: string;
    onFilePrepared: () => void;
};

export const ImportProcess = ({ stagedDatasetId, onFilePrepared }: ImportProcessProps) => {
    const { getImportEntry, deleteImportEntry, updateImportEntryStep } = useImportDatasetAsNewProject();
    const entry = getImportEntry(stagedDatasetId);

    return (
        <ImportJobProcess
            jobId={entry?.prepareJobId}
            fileName={entry?.fileName ?? ''}
            onError={() => {
                deleteImportEntry(stagedDatasetId);
            }}
            onSuccess={() => {
                onFilePrepared();
                updateImportEntryStep(stagedDatasetId, 'taskTypeSelection');
            }}
            message='Prepare dataset import as new project'
        />
    );
};
