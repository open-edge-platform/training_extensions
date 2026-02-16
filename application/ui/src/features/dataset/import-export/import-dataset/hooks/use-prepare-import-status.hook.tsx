// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { toast } from '@geti/ui';
import { isEmpty, isFunction } from 'lodash-es';

import { $api } from '../../../../../api/client';
import { PrepareImportDatasetJob } from '../../../../../constants/shared-types';
import { useLabelMappingImportDataset } from '../../../../../hooks/localStorage/use-label-mapping-import-dataset.hook';
import { usePrepareImportDataset } from '../../../../../hooks/localStorage/use-prepare-import-dataset.hook';
import { isInvalidJob, isJobDone, isJobFailed } from '../../util';

type UsePrepareImportStatusProps = {
    onError?: () => void;
    onSuccess?: () => void;
};

export const usePrepareImportStatus = ({ onError, onSuccess }: UsePrepareImportStatusProps) => {
    const { getLsPreparingImport, removeLsPreparingImport } = usePrepareImportDataset();
    const { addLsLabelMappingImport, getLsLabelMappingImport } = useLabelMappingImportDataset();
    const { id: jobId, fileName, size } = getLsPreparingImport() ?? {};

    const response = $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        { params: { path: { job_id: jobId } } },
        {
            enabled: !isEmpty(jobId),
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
            removeLsPreparingImport();
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.error?.detail}` });
        }
    }, [onError, removeLsPreparingImport, response.error, response.isError]);

    useEffect(() => {
        if (isJobFailed(response.data)) {
            isFunction(onError) && onError();
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.data?.message}` });
        }
    }, [onError, response.data]);

    useEffect(() => {
        if (isJobDone(response.data) && getLsLabelMappingImport() === null) {
            addLsLabelMappingImport(stagedDatasetId);
            isFunction(onSuccess) && onSuccess();
        }
    }, [onSuccess, getLsLabelMappingImport, addLsLabelMappingImport, stagedDatasetId, response]);

    return { ...response, fileName, size };
};
