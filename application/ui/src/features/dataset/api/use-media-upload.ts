// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../api/client';
import { getQueryKey } from '../../../query-client/query-client';

export const MEDIA_UPLOAD_CONCURRENCY = 5;

// Runs ${batchSize} promises at a time until all promises have been executed,
// returning an array of settled results.
const executeInBatches = async <T>(
    promises: Array<() => Promise<T>>,
    batchSize: number,
    onBatchCompleted?: () => Promise<void>
) => {
    const settledResults: PromiseSettledResult<T>[] = [];

    for (let index = 0; index < promises.length; index += batchSize) {
        const batchPromises = promises.slice(index, index + batchSize).map((promise) => promise());
        const batchResults = await Promise.allSettled(batchPromises);

        settledResults.push(...batchResults);

        if (onBatchCompleted !== undefined) {
            await onBatchCompleted();
        }
    }

    return settledResults;
};

export const useMediaUpload = () => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();

    const addItemMutation = $api.useMutation('post', '/api/projects/{project_id}/dataset/media');
    const invalidateMediaQuery = () =>
        queryClient.invalidateQueries({
            queryKey: getQueryKey([
                'get',
                '/api/projects/{project_id}/dataset/media',
                { params: { path: { project_id: projectId } } },
            ]),
        });

    const uploadMedia = async (files: File[]) => {
        const uploadPromises = files.map((file) => {
            return () => {
                const formData = new FormData();
                formData.append('file', file);

                return addItemMutation.mutateAsync({
                    params: { path: { project_id: projectId } },
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    body: formData as any,
                });
            };
        });

        const settledResults = await executeInBatches(uploadPromises, MEDIA_UPLOAD_CONCURRENCY, invalidateMediaQuery);

        const succeeded = settledResults.filter((result) => result.status === 'fulfilled').length;
        const failed = settledResults.filter((result) => result.status === 'rejected').length;

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
        uploadMedia,
        isUploading: addItemMutation.isPending,
    };
};
