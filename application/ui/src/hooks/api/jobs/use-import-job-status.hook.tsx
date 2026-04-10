// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { toast } from '@geti/ui';
import { isFunction } from 'lodash-es';

import { $api } from '../../../api/client';
import { isNonEmptyString } from '../../../shared/util';
import { isInvalidJob, isJobDone, isJobFailed } from '../util';

type UseImportJobStatusProps = {
    jobId: string | null | undefined;
    onError?: (error: unknown) => void;
    onSuccess?: () => void;
};

export const useImportJobStatus = ({ jobId, onError, onSuccess }: UseImportJobStatusProps) => {
    const response = $api.useQuery(
        'get',
        '/api/jobs/{job_id}',
        { params: { path: { job_id: jobId } } },
        {
            enabled: isNonEmptyString(jobId),
            refetchInterval: ({ state }) => {
                return isJobDone(state.data) || isJobFailed(state.data) || state.status === 'error' ? false : 1_000;
            },
        }
    );

    useEffect(() => {
        if (response.isError && isInvalidJob(response.error)) {
            isFunction(onError) && onError(response.error);
            toast({ type: 'error', message: `An error occurred during import. ${response.error?.detail}` });
        }
    }, [onError, response.error, response.isError]);

    useEffect(() => {
        if (isJobFailed(response.data)) {
            isFunction(onError) && onError(response.error);
            toast({ type: 'error', message: `An error occurred during import. ${response.data?.message}` });
        }
    }, [onError, response.data, response.error]);

    useEffect(() => {
        if (isJobDone(response.data)) {
            isFunction(onSuccess) && onSuccess();
        }
    }, [onSuccess, response.data]);

    return response;
};
