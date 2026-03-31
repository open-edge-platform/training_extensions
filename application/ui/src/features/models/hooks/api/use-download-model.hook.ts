// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import { ModelVariant } from '../../../../constants/shared-types';

export const useDownloadModel = (modelId: string, format: ModelVariant['format']) => {
    const projectId = useProjectIdentifier();

    const mutation = $api.useMutation(
        'get',
        '/api/projects/{project_id}/models/{model_id}/variants/{model_variant_id}/binary',
        {
            onSuccess: (data, variables) => {
                const blob = data as Blob;
                const url = URL.createObjectURL(blob);
                const modelVariantId = variables.params.path?.model_variant_id;

                const link = document.createElement('a');
                link.href = url;
                link.download = modelVariantId ? `${format}-${modelVariantId}.zip` : `${format}-${modelId}.zip`;
                link.click();

                URL.revokeObjectURL(url);

                toast({ type: 'success', message: 'Model downloaded successfully' });
            },
            onError: () => {
                toast({ type: 'error', message: 'Failed to download model' });
            },
        }
    );

    const downloadModel = (modelVariantId: string) => {
        mutation.mutate({
            params: {
                path: { project_id: projectId, model_id: modelId, model_variant_id: modelVariantId },
            },
            parseAs: 'blob',
        });
    };

    return {
        downloadModel,
        isDownloading: mutation.isPending,
        error: mutation.error,
    };
};
