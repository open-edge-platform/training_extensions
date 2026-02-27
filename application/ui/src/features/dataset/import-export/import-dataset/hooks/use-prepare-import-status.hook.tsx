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
    stagedDatasetId: string;
    onError?: () => void;
    onSuccess?: () => void;
};

export const usePrepareImportStatus = ({ stagedDatasetId, onError, onSuccess }: UsePrepareImportStatusProps) => {
    const { findImportEntry, updateImportEntryStep, deleteImportEntry } = useImportDatasetToProject();
    const importLsEntry = findImportEntry({ stagedDatasetId });

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

    useEffect(() => {
        if (response.isError && isInvalidJob(response.error)) {
            isFunction(onError) && onError();
            deleteImportEntry(stagedDatasetId);
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.error?.detail}` });
        }
    }, [deleteImportEntry, onError, response.error, response.isError, stagedDatasetId]);

    useEffect(() => {
        if (isJobFailed(response.data)) {
            isFunction(onError) && onError();
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.data?.message}` });
        }
    }, [onError, response.data]);

    useEffect(() => {
        if (isJobDone(response.data)) {
            isFunction(onSuccess) && onSuccess();
            updateImportEntryStep(stagedDatasetId, 'labelMapping');
        }
    }, [onSuccess, updateImportEntryStep, response.data, stagedDatasetId]);

    return {
        ...response,
        size: importLsEntry?.size ?? 0,
        fileName: importLsEntry?.fileName ?? '',
        stagedDatasetId,
    };
};
