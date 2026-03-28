// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isFunction } from 'lodash-es';

import { $api } from '../../../api/client';
import { getQueryKey } from '../../../query-client/query-client';
import { useUploadProgress } from '../hooks/use-display-upload-progress';

export const MEDIA_UPLOAD_CONCURRENCY = 10;

type UploadTask = () => Promise<unknown>;

// Runs ${batchSize} promises at a time until all promises have been executed,
// returning an array of settled results.
const executeInBatches = async (
    uploadTasks: UploadTask[],
    batchSize: number,
    onBatchCompleted?: (batchResults: PromiseSettledResult<unknown>[]) => Promise<void>
): Promise<PromiseSettledResult<unknown>[]> => {
    const settledResults: PromiseSettledResult<unknown>[] = [];

    for (let index = 0; index < uploadTasks.length; index += batchSize) {
        const batchPromises = uploadTasks.slice(index, index + batchSize).map((task) => task());
        const batchResults = await Promise.allSettled(batchPromises);

        settledResults.push(...batchResults);

        if (isFunction(onBatchCompleted)) {
            await onBatchCompleted(batchResults);
        }
    }

    return settledResults;
};

export const useMediaUpload = () => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();
    const { uploadProgress, startUploadProgress, updateUploadProgress, finishUploadProgress } = useUploadProgress();

    const addItemMutation = $api.useMutation('post', '/api/projects/{project_id}/dataset/media');
    type UploadMutationRequest = Parameters<typeof addItemMutation.mutateAsync>[0];

    const buildUploadTask = (file: File): UploadTask => {
        return () => {
            const formData = new FormData();
            formData.append('file', file);

            const request: UploadMutationRequest = {
                params: { path: { project_id: projectId } },
                body: formData as unknown as UploadMutationRequest['body'],
            };

            return addItemMutation.mutateAsync(request);
        };
    };

    const invalidateMediaQuery = () =>
        queryClient.invalidateQueries({
            queryKey: getQueryKey([
                'get',
                '/api/projects/{project_id}/dataset/media',
                { params: { path: { project_id: projectId } } },
            ]),
        });

    // Processes files with batched concurrency
    const processUploadBatch = async (files: File[]): Promise<void> => {
        if (files.length === 0) {
            return;
        }

        startUploadProgress(files.length);

        try {
            const onBatchCompleted = async (batchResults: PromiseSettledResult<unknown>[]) => {
                updateUploadProgress({ settledResults: batchResults });

                await invalidateMediaQuery();
            };

            const uploadTasks = files.map((file) => buildUploadTask(file));
            await executeInBatches(uploadTasks, MEDIA_UPLOAD_CONCURRENCY, onBatchCompleted);

            finishUploadProgress();
        } catch (_error) {
            finishUploadProgress();
        }
    };

    // Starts the upload process directly
    const uploadMedia = (files: File[]): void => {
        if (files.length === 0) {
            return;
        }

        void processUploadBatch(files);
    };

    return {
        uploadMedia,
        uploadProgress,
    };
};
