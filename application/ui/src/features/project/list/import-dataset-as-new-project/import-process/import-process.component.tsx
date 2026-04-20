// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useImportDatasetAsNewProject } from 'hooks/storage/use-import-dataset-as-new-project.hook';

import { ImportJobProcess } from '../../../../../components/import-job-process/import-job-process.component';
import { useImportDatasetDialog } from '../../../providers/import-dataset-dialog-provider.component';

type ImportProcessProps = {
    stagedDatasetId: string;
    onFilePrepared: () => void;
};

export const ImportProcess = ({ stagedDatasetId, onFilePrepared }: ImportProcessProps) => {
    const { datasetImportDialogState } = useImportDatasetDialog();
    const { getImportEntry, updateImportEntryStep } = useImportDatasetAsNewProject();

    const entry = getImportEntry(stagedDatasetId);

    return (
        <ImportJobProcess
            jobId={entry?.prepareJobId}
            fileName={entry?.fileName ?? ''}
            onError={datasetImportDialogState.close}
            onSuccess={() => {
                onFilePrepared();
                updateImportEntryStep(stagedDatasetId, 'taskTypeSelection');
            }}
            message='Prepare dataset import as new project'
        />
    );
};
