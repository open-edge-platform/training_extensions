// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../api/client';
import { getQueryKey } from '../../../query-client/query-client';

export const useMediaUpload = () => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();

    const addItemMutation = $api.useMutation('post', '/api/projects/{project_id}/dataset/media');

    const uploadMedia = async (files: File[]) => {
        const uploadPromises = files.map((file) => {
            const formData = new FormData();
            formData.append('file', file);

            return addItemMutation.mutateAsync({
                params: { path: { project_id: projectId } },
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                body: formData as any,
            });
        });

        const promises = await Promise.allSettled(uploadPromises);

        const succeeded = promises.filter((result) => result.status === 'fulfilled').length;
        const failed = promises.filter((result) => result.status === 'rejected').length;

        await queryClient.invalidateQueries({
            queryKey: getQueryKey([
                'get',
                '/api/projects/{project_id}/dataset/media',
                { params: { path: { project_id: projectId } } },
            ]),
        });

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
