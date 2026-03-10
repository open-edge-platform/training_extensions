// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { toast } from '@geti/ui';

export type UploadProgress = {
    total: number;
    completed: number;
    succeeded: number;
    failed: number;
    isUploading: boolean;
};

export type UploadOutcome = Pick<UploadProgress, 'succeeded' | 'failed'>;

type UploadProgressUpdate = {
    settledResults: PromiseSettledResult<unknown>[];
};
type UploadProgressFinish = UploadProgressUpdate;

const INITIAL_UPLOAD_PROGRESS: UploadProgress = {
    total: 0,
    completed: 0,
    succeeded: 0,
    failed: 0,
    isUploading: false,
};

const summarizeSettledResults = <T>(results: PromiseSettledResult<T>[]): UploadOutcome => {
    const succeeded = results.filter((result) => result.status === 'fulfilled').length;
    const failed = results.filter((result) => result.status === 'rejected').length;

    return { succeeded, failed };
};

export const useUploadProgress = () => {
    const [uploadProgress, setUploadProgress] = useState<UploadProgress>(INITIAL_UPLOAD_PROGRESS);

    const startUploadProgress = (total: number): void => {
        setUploadProgress({ total, completed: 0, succeeded: 0, failed: 0, isUploading: true });
    };

    const updateUploadProgress = ({ settledResults }: UploadProgressUpdate): void => {
        const { succeeded, failed } = summarizeSettledResults(settledResults);

        setUploadProgress((previousProgress) => {
            const nextCompleted = previousProgress.completed + settledResults.length;
            const total = previousProgress.total;

            return {
                total,
                completed: nextCompleted,
                succeeded: previousProgress.succeeded + succeeded,
                failed: previousProgress.failed + failed,
                isUploading: nextCompleted < total,
            };
        });
    };

    const finishUploadProgress = ({ settledResults }: UploadProgressFinish): void => {
        const { succeeded, failed } = summarizeSettledResults(settledResults);

        setUploadProgress((previousProgress) => ({
            total: previousProgress.total,
            completed: previousProgress.total,
            succeeded,
            failed,
            isUploading: false,
        }));

        if (failed === 0) {
            toast({ type: 'success', message: `Uploaded ${succeeded} item(s)` });
        } else if (succeeded === 0) {
            toast({ type: 'error', message: `Failed to upload ${failed} item(s)` });
        } else {
            toast({
                type: 'warning',
                message: `Uploaded ${succeeded} item(s), ${failed} failed`,
            });
        }
    };

    return {
        uploadProgress,
        startUploadProgress,
        updateUploadProgress,
        finishUploadProgress,
    };
};
