// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { useMutation } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { fetchClient } from '../../../../api/client';
import { downloadFile } from '../../../../shared/util';

export const useDownloadModel = (modelId: string) => {
    const projectId = useProjectIdentifier();

    const mutation = useMutation({
        mutationFn: async (modelVariantId: string) => {
            const { data, error, response } = await fetchClient.GET(
                '/api/projects/{project_id}/models/{model_id}/variants/{model_variant_id}/binary',
                {
                    params: { path: { project_id: projectId, model_id: modelId, model_variant_id: modelVariantId } },
                    parseAs: 'blob',
                }
            );

            if (error || !data) {
                throw new Error(`Failed to download model: ${response.status} ${response.statusText}`);
            }

            const contentDisposition = response.headers.get('content-disposition');
            const filename = contentDisposition?.match(/filename="?([^"]+)"?/)?.[1] ?? `model-${modelId}.zip`;

            const url = URL.createObjectURL(data);
            downloadFile(url, filename);
        },
        onSuccess: () => {
            toast({ type: 'success', message: 'Model downloaded successfully' });
        },
        onError: () => {
            toast({ type: 'error', message: 'Failed to download model' });
        },
    });

    const downloadModel = (modelVariantId: string) => {
        mutation.mutate(modelVariantId);
    };

    return {
        downloadModel,
        isDownloading: mutation.isPending,
        error: mutation.error,
    };
};
