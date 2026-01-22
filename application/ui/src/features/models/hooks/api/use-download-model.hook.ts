// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import type { ModelFormat } from '../../../../constants/shared-types';

export const useDownloadModel = (modelId: string) => {
    const projectId = useProjectIdentifier();

    const mutation = $api.useMutation('get', '/api/projects/{project_id}/models/{model_id}/binary', {
        onSuccess: (data, variables) => {
            const blob = data as Blob;
            const url = URL.createObjectURL(blob);
            const format = variables.params.query?.format;

            const link = document.createElement('a');

            link.href = url;
            link.download = format ? `model-${modelId}-${format}.zip` : `model-${modelId}.zip`;

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            URL.revokeObjectURL(url);

            toast({ type: 'success', message: 'Model downloaded successfully' });
        },
        onError: () => {
            toast({ type: 'error', message: 'Failed to download model' });
        },
    });

    const downloadModel = (format?: ModelFormat) => {
        mutation.mutate({
            params: {
                path: { project_id: projectId, model_id: modelId },
                query: format ? { format } : undefined,
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
