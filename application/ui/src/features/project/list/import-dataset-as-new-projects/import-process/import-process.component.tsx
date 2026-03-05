// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { useImportDatasetAsNewProject } from 'hooks/localStorage/use-import-dataset-as-new-project.hook';

import { ImportJobProcess } from '../../../../../components/import-job-process/import-job-process.component';
import { ImportDatasetAsNewProjectState } from '../../../../dataset/import-export/import-dataset/util';

type ImportProcessProps = {
    stagedDatasetId: string;
    setCurrentStep: Dispatch<SetStateAction<ImportDatasetAsNewProjectState>>;
};

export const ImportProcess = ({ stagedDatasetId, setCurrentStep }: ImportProcessProps) => {
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
                setCurrentStep('labelMapping');
                updateImportEntryStep(stagedDatasetId, 'labelMapping');
            }}
            message='Prepare dataset import as new project'
        />
    );
};
