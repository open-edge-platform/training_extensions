// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { toast } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useImportDatasetToProject } from 'hooks/localStorage/use-import-dataset-to-project.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../../api/client';
import { PrepareImportDatasetJob } from '../../../../../constants/shared-types';
import { getQueryKey } from '../../../../../query-client/query-client';
import { isNonEmptyString } from '../../../../../shared/util';
import { isInvalidJob, isJobDone, isJobFailed } from '../../util';

type UseImportStatusProps = {
    stagedDatasetId: string;
};

export const useImportStatus = ({ stagedDatasetId }: UseImportStatusProps) => {
    const queryClient = useQueryClient();
    const { getImportEntry, deleteImportEntry } = useImportDatasetToProject();
    const importLsEntry = getImportEntry({ stagedDatasetId });
    const projectId = useProjectIdentifier();

    const response = $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        { params: { path: { job_id: importLsEntry?.importJobId } } },
        {
            enabled: isNonEmptyString(importLsEntry?.importJobId),
            select: (currentData) => currentData as PrepareImportDatasetJob,
            refetchInterval: ({ state }) => {
                return isJobDone(state.data) || isJobFailed(state.data) || state.status === 'error' ? false : 1_000;
            },
        }
    );

    useEffect(() => {
        if (response.isError && isInvalidJob(response.error)) {
            deleteImportEntry(stagedDatasetId);
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.error?.detail}` });
        }
    }, [deleteImportEntry, response.error, response.isError, stagedDatasetId]);

    useEffect(() => {
        if (isJobFailed(response.data)) {
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.data?.message}` });
        }
    }, [response.data]);

    useEffect(() => {
        if (isJobDone(response.data)) {
            queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/projects/{project_id}/dataset/media',
                    { params: { path: { project_id: projectId } } },
                ]),
            });
        }
    }, [queryClient, response.data, stagedDatasetId, projectId]);

    return {
        ...response,
        size: importLsEntry?.size ?? 0,
        fileName: importLsEntry?.fileName ?? '',
        stagedDatasetId,
    };
};
