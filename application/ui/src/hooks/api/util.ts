// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import isObject from 'lodash-es/isObject';

import { Job } from '../../constants/shared-types';

export const isInvalidStagedFile = (error: unknown): boolean => {
    if (isObject(error) && 'detail' in error) {
        const detail = String(error.detail);
        return detail.includes('not found');
    }

    return false;
};

export const getJobProgress = (progress?: number) => Math.round(Math.max(0, Math.min(100, progress ?? 0)));

export const isInvalidJob = (error: unknown): boolean => {
    if (isObject(error) && 'detail' in error) {
        const detail = String(error.detail);
        return detail.includes('Job not found') || detail.includes('Invalid job_id');
    }

    return false;
};

export const isJobDone = (job?: Job): job is Job => job?.status === 'DONE';
export const isJobFailed = (job?: Job): job is Job => job?.status === 'FAILED';
export const isJobRunning = (job?: Job): job is Job => job?.status === 'RUNNING';
export const isJobPending = (job?: Job): job is Job => job?.status === 'PENDING';

export const isTrainJob = (job?: Job): job is Job => job?.job_type === 'train';
export const isQuantizeJob = (job?: Job): job is Job => job?.job_type === 'quantize';
