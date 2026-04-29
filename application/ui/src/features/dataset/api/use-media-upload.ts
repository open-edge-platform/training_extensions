// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isFunction } from 'lodash-es';

import { $api } from '../../../api/client';
import { MediaDTO } from '../../../constants/shared-types';
import { getQueryKey } from '../../../query-client/query-client';
import { useUploadProgress } from '../hooks/use-display-upload-progress';

export const MEDIA_UPLOAD_CONCURRENCY = 10;

type UploadTask<T> = () => Promise<T>;

// Runs ${batchSize} promises at a time until all promises have been executed,
// returning an array of settled results.
const executeInBatches = async <T>(
    uploadTasks: UploadTask<T>[],
    batchSize: number,
    onBatchCompleted?: (batchResults: PromiseSettledResult<T>[]) => Promise<void>
): Promise<PromiseSettledResult<T>[]> => {
    const settledResults: PromiseSettledResult<T>[] = [];

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

const getFulfilledValues = <T>(results: PromiseSettledResult<T>[]): T[] =>
    results
        .filter((result): result is PromiseFulfilledResult<T> => result.status === 'fulfilled')
        .map((result) => result.value);

export const useMediaUpload = () => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();
    const { uploadProgress, startUploadProgress, updateUploadProgress, finishUploadProgress } = useUploadProgress();

    const addItemMutation = $api.useMutation('post', '/api/projects/{project_id}/dataset/media');
    type UploadMutationRequest = Parameters<typeof addItemMutation.mutateAsync>[0];

    const buildUploadTask = (file: File): UploadTask<MediaDTO> => {
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

    const invalidateMediaQuery = () => {
        return Promise.all([
            queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/projects/{project_id}/dataset/media',
                    { params: { path: { project_id: projectId } } },
                ]),
            }),
            queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/projects/{project_id}/dataset/statistics',
                    { params: { path: { project_id: projectId } } },
                ]),
            }),
        ]);
    };

    // Processes files with batched concurrency, returning all successfully uploaded media items
    const processUploadBatch = async (files: File[]): Promise<MediaDTO[]> => {
        if (files.length === 0) {
            return [];
        }

        startUploadProgress(files.length);

        try {
            const onBatchCompleted = async (batchResults: PromiseSettledResult<MediaDTO>[]) => {
                updateUploadProgress({ settledResults: batchResults });

                await invalidateMediaQuery();
            };

            const uploadTasks = files.map((file) => buildUploadTask(file));
            const allResults = await executeInBatches(uploadTasks, MEDIA_UPLOAD_CONCURRENCY, onBatchCompleted);

            finishUploadProgress();

            return getFulfilledValues(allResults);
        } catch (_error) {
            finishUploadProgress();
            return [];
        }
    };

    // Starts the upload process directly, returning all successfully uploaded media items
    const uploadMedia = async (files: File[]): Promise<MediaDTO[]> => {
        if (files.length === 0) {
            return [];
        }

        return processUploadBatch(files);
    };

    return {
        uploadMedia,
        uploadProgress,
    };
};
