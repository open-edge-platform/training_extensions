// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isObject } from 'lodash-es';

import { Job } from '../../../../constants/shared-types';

export const isInvalidJob = (error: unknown): boolean =>
    isObject(error) && 'detail' in error && error.detail === 'Job not found';

export const isJobDone = (job?: Job) => job?.status === 'DONE';
export const isJobFailed = (job?: Job) => job?.status === 'FAILED';
export const isJobRunning = (job?: Job) => job?.status === 'RUNNING';
export const isJobPending = (job?: Job) => job?.status === 'PENDING';
