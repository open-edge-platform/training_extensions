// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { toast } from '@geti/ui';

type UploadProgress = {
    total: number;
    completed: number;
    succeeded: number;
    failed: number;
    isUploading: boolean;
};

type UploadOutcome = Pick<UploadProgress, 'succeeded' | 'failed'>;

type UploadProgressUpdate = {
    settledResults: PromiseSettledResult<unknown>[];
};

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

const UPLOAD_TOAST_ID = 'upload-progress-notification';

export const useUploadProgress = () => {
    const [uploadProgress, setUploadProgress] = useState<UploadProgress>(INITIAL_UPLOAD_PROGRESS);

    const startUploadProgress = (total: number): void => {
        setUploadProgress(() => ({
            total,
            completed: 0,
            succeeded: 0,
            failed: 0,
            isUploading: true,
        }));
    };

    const updateUploadProgress = ({ settledResults }: UploadProgressUpdate): void => {
        const { succeeded, failed } = summarizeSettledResults(settledResults);

        setUploadProgress((previousProgress) => {
            const nextCompleted = previousProgress.completed + settledResults.length;
            const total = previousProgress.total;
            const nextSucceeded = previousProgress.succeeded + succeeded;
            const nextFailed = previousProgress.failed + failed;

            const msg = `Uploading ${total} item(s)... (${nextSucceeded} succeeded, ${nextFailed} failed)`;

            toast({
                id: UPLOAD_TOAST_ID,
                type: 'info',
                message: msg,
            });

            return {
                total,
                completed: nextCompleted,
                succeeded: nextSucceeded,
                failed: nextFailed,
                isUploading: true,
            };
        });
    };

    const finishUploadProgress = (): void => {
        setUploadProgress((previousProgress) => {
            const nextProgress = {
                ...previousProgress,
                isUploading: false,
            };

            let toastType: 'success' | 'error' | 'warning' = 'success';
            let msg = `Uploaded ${nextProgress.succeeded} item(s)`;

            if (nextProgress.failed === 0) {
                toastType = 'success';
                msg = `Uploaded ${nextProgress.succeeded} item(s)`;
            } else if (nextProgress.succeeded === 0) {
                toastType = 'error';
                msg = `Failed to upload ${nextProgress.failed} item(s)`;
            } else {
                toastType = 'warning';
                msg = `Uploaded ${nextProgress.succeeded} item(s), ${nextProgress.failed} failed`;
            }

            toast({
                id: UPLOAD_TOAST_ID,
                type: toastType,
                message: msg,
                duration: 5000,
            });

            return nextProgress;
        });
    };

    return {
        uploadProgress,
        startUploadProgress,
        updateUploadProgress,
        finishUploadProgress,
    };
};
