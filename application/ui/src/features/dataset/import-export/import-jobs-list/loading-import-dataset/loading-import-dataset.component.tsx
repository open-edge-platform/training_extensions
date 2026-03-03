// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { useImportDatasetToProject } from '../../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import { getQueryKey } from '../../../../../query-client/query-client';
import { useImportJobStatus } from '../../import-dataset/hooks/use-import-job-status.hook';
import { isJobDone, isJobFailed, isJobPending, isJobRunning } from '../../util';
import { ImportActiveJob } from '../import-active-job/import-active-job.component';
import { ImportFailedJob } from '../import-failed-job/import-failed-job.component';
import { ImportJobDone } from '../import-job-done/import-job-done.component';

type LoadingImportDatasetProps = {
    stagedDatasetId: string;
};

export const LoadingImportDataset = ({ stagedDatasetId }: LoadingImportDatasetProps) => {
    const queryClient = useQueryClient();
    const projectId = useProjectIdentifier();

    const { getImportEntry } = useImportDatasetToProject();
    const importLsEntry = getImportEntry(stagedDatasetId);

    const { data: job } = useImportJobStatus({
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
                <ImportFailedJob job={job} fileName={fileName} size={size} stagedDatasetId={stagedDatasetId} />
            )}

            {isRunningOrPending && (
                <ImportActiveJob job={job} fileName={fileName} size={size} stagedDatasetId={stagedDatasetId} />
            )}

            {isJobDone(job) && <ImportJobDone fileName={fileName} size={size} stagedDatasetId={stagedDatasetId} />}
        </View>
    );
};
