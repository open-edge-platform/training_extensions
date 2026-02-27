// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { toast } from '@geti/ui';
import { useImportDatasetToProject } from 'hooks/localStorage/use-import-dataset-to-project.hook';
import { isFunction } from 'lodash-es';

import { $api } from '../../../../../api/client';
import { PrepareImportDatasetJob } from '../../../../../constants/shared-types';
import { isNonEmptyString } from '../../../../../shared/util';
import { isInvalidJob, isJobDone, isJobFailed } from '../../util';

type UsePrepareImportStatusProps = {
    prepareJobId: string;
    onError?: () => void;
    onSuccess?: (stagedDatasetId: string) => void;
};

export const usePrepareImportStatus = ({ prepareJobId, onError, onSuccess }: UsePrepareImportStatusProps) => {
    const { findImportEntry, updateImportEntryStagedId, deleteImportEntry } = useImportDatasetToProject();
    const importLsEntry = findImportEntry({ prepareJobId });

    const response = $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        { params: { path: { job_id: importLsEntry?.prepareJobId } } },
        {
            enabled: isNonEmptyString(importLsEntry?.prepareJobId),
            select: (currentData) => currentData as PrepareImportDatasetJob,
            refetchInterval: ({ state }) => {
                return isJobDone(state.data) || isJobFailed(state.data) || state.status === 'error' ? false : 1_000;
            },
        }
    );

    const stagedDatasetId = response.data?.metadata.staged_dataset_id ?? '';

    useEffect(() => {
        if (response.isError && isInvalidJob(response.error)) {
            isFunction(onError) && onError();
            deleteImportEntry({ prepareJobId: importLsEntry?.prepareJobId });
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.error?.detail}` });
        }
    }, [onError, deleteImportEntry, response, importLsEntry?.prepareJobId]);

    useEffect(() => {
        if (isJobFailed(response.data)) {
            isFunction(onError) && onError();
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.data?.message}` });
        }
    }, [onError, response.data]);

    useEffect(() => {
        if (isJobDone(response.data)) {
            updateImportEntryStagedId(prepareJobId, stagedDatasetId);
            isFunction(onSuccess) && onSuccess(stagedDatasetId);
        }
    }, [onSuccess, updateImportEntryStagedId, prepareJobId, stagedDatasetId, response.data]);

    return { ...response, fileName: importLsEntry?.fileName, size: importLsEntry?.size ?? 0 };
};
