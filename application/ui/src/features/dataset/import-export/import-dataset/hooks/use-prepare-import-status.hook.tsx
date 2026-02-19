// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { isEmpty } from 'lodash';

import { toast } from '../../../../../../packages/ui/src/toast/toast.component';
import { $api } from '../../../../../api/client';
import { PrepareImportDatasetJob } from '../../../../../constants/shared-types';
import { usePrepareImportDataset } from '../../../../../hooks/localStorage/use-prepare-import-dataset.hook';
import { isInvalidJob, isJobDone, isJobFailed } from '../../export-jobs-list/util';

type UsePrepareImportStatusProps = {
    onError: () => void;
};

export const usePrepareImportStatus = ({ onError }: UsePrepareImportStatusProps) => {
    const { getLsPreparingImportId } = usePrepareImportDataset();
    const { id: jobId, fileName } = getLsPreparingImportId() ?? {};

    const { removeLsPreparingImportId } = usePrepareImportDataset();

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

    useEffect(() => {
        if (response.isError && isInvalidJob(response.error)) {
            onError();
            removeLsPreparingImportId();
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.error?.detail}` });
        }

        if (isJobFailed(response.data)) {
            onError();
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.data?.message}` });
        }
    }, [onError, removeLsPreparingImportId, response.data, response.error, response.isError]);

    return { ...response, fileName };
};
