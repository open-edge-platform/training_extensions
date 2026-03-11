// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useImportJobStatus } from 'hooks/api/jobs/use-import-job-status.hook';
import { isJobDone, isJobFailed, isJobPending, isJobRunning } from 'hooks/api/util';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { ImportActiveJob } from '../../../../../components/prepare-import-dataset/import-active-job/import-active-job.component';
import { ImportFailedJob } from '../../../../../components/prepare-import-dataset/import-failed-job/import-failed-job.component';
import { useImportDatasetToProject } from '../../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import { getQueryKey } from '../../../../../query-client/query-client';
import { ImportJobDone } from '../import-job-done/import-job-done.component';

type LoadingImportDatasetProps = {
    stagedDatasetId: string;
};

export const LoadingImportDataset = ({ stagedDatasetId }: LoadingImportDatasetProps) => {
    const queryClient = useQueryClient();
    const projectId = useProjectIdentifier();

    const { getImportEntry, deleteImportEntry } = useImportDatasetToProject();
    const importLsEntry = getImportEntry(stagedDatasetId);

    const {
        error,
        isError,
        data: job,
    } = useImportJobStatus({
        jobId: importLsEntry?.importJobId,
        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/projects/{project_id}/dataset/media',
                    { params: { path: { project_id: projectId } } },
                ]),
            });
        },
    });

    const size = importLsEntry?.size ?? 0;
    const fileName = importLsEntry?.fileName ?? '';
    const isRunningOrPending = isJobRunning(job) || isJobPending(job);

    return (
        <View
            position='relative'
            borderColor='gray-200'
            borderRadius='regular'
            backgroundColor='gray-75'
            borderWidth='thin'
        >
            {isJobFailed(job) && (
                <ImportFailedJob
                    fileName={fileName}
                    size={size}
                    error={job.error ?? ''}
                    message={job.message ?? ''}
                    stagedDatasetId={stagedDatasetId}
                    deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                />
            )}

            {isError && (
                <ImportFailedJob
                    size={size}
                    fileName={fileName}
                    error={`${error?.detail ?? 'Unknown error'}`}
                    message={'An error occurred during import.'}
                    stagedDatasetId={stagedDatasetId}
                    deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                />
            )}

            {isRunningOrPending && (
                <ImportActiveJob
                    job={job}
                    fileName={fileName}
                    size={size}
                    stagedDatasetId={stagedDatasetId}
                    deleteEntry={() => deleteImportEntry(stagedDatasetId)}
                />
            )}

            {isJobDone(job) && <ImportJobDone fileName={fileName} size={size} stagedDatasetId={stagedDatasetId} />}
        </View>
    );
};
