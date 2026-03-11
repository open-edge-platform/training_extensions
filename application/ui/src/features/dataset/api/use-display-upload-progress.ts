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

const UPLOAD_TOAST_ID = 'upload-progress-notification';

export const useUploadProgress = () => {
    const [uploadProgress, setUploadProgress] = useState<UploadProgress>(INITIAL_UPLOAD_PROGRESS);

    const startUploadProgress = (total: number): void => {
        setUploadProgress({ total, completed: 0, succeeded: 0, failed: 0, isUploading: true });

        toast({
            id: UPLOAD_TOAST_ID,
            type: 'info',
            message: `Uploading ${total} item(s)... (0 succeeded, 0 failed)`,
            duration: 0,
        });
    };

    const updateUploadProgress = ({ settledResults }: UploadProgressUpdate): void => {
        const { succeeded, failed } = summarizeSettledResults(settledResults);

        setUploadProgress((previousProgress) => {
            const nextCompleted = previousProgress.completed + settledResults.length;
            const total = previousProgress.total;
            const totalSucceeded = previousProgress.succeeded + succeeded;
            const totalFailed = previousProgress.failed + failed;
            const isUploading = nextCompleted < total;

            const progressMessage = `
                Uploading ${total} item(s)... (${totalSucceeded} succeeded, ${totalFailed} failed)
            `;

            toast({
                id: UPLOAD_TOAST_ID,
                type: 'info',
                message: progressMessage,
                duration: 0,
            });

            return {
                total,
                completed: nextCompleted,
                succeeded: totalSucceeded,
                failed: totalFailed,
                isUploading,
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
            toast({
                id: UPLOAD_TOAST_ID,
                type: 'success',
                message: `Uploaded ${succeeded} item(s)`,
                duration: 3000,
            });
        } else if (succeeded === 0) {
            toast({
                id: UPLOAD_TOAST_ID,
                type: 'error',
                message: `Failed to upload ${failed} item(s)`,
                duration: 3000,
            });
        } else {
            toast({
                id: UPLOAD_TOAST_ID,
                type: 'warning',
                message: `Uploaded ${succeeded} item(s), ${failed} failed`,
                duration: 3000,
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
