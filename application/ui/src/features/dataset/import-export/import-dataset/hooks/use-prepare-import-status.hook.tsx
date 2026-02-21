// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { toast } from '@geti/ui';
import { isEmpty, isFunction } from 'lodash-es';

import { $api } from '../../../../../api/client';
import { PrepareImportDatasetJob } from '../../../../../constants/shared-types';
import { usePrepareImportDataset } from '../../../../../hooks/localStorage/use-prepare-import-dataset.hook';
import { isInvalidJob, isJobDone, isJobFailed } from '../../util';

type UsePrepareImportStatusProps = {
    onError: () => void;
};

export const usePrepareImportStatus = ({ onError }: UsePrepareImportStatusProps) => {
    const { getLsPreparingImportId, removeLsPreparingImportId } = usePrepareImportDataset();
    const { id: jobId, fileName, size } = getLsPreparingImportId() ?? {};

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
            isFunction(onError) && onError();
            removeLsPreparingImportId();
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.error?.detail}` });
        }
    }, [onError, removeLsPreparingImportId, response.error, response.isError]);

    useEffect(() => {
        if (isJobFailed(response.data)) {
            isFunction(onError) && onError();
            toast({ type: 'error', message: `Failed to prepare dataset for import. ${response.data?.message}` });
        }
    }, [onError, response.data]);

    return { ...response, fileName, size };
};
