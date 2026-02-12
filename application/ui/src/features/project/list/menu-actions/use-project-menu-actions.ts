// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key, toast } from '@geti/ui';

import { $api } from '../../../../api/client';
import { useDisablePipeline, useEnablePipeline } from '../../../../hooks/api/pipeline.hook';

const PROJECT_ACTIONS = { rename: 'Rename', delete: 'Delete' };

type ProjectMenuCallbacks = {
    onRename: () => void;
    onDelete: () => void;
};

export const useProjectMenuActions = (projectId: string, callbacks: ProjectMenuCallbacks) => {
    const { data: pipeline } = $api.useQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });

    const enablePipelineMutation = useEnablePipeline();
    const disablePipelineMutation = useDisablePipeline();

    const isPipelineRunning = pipeline?.status === 'running';

    const menuActions = {
        ...(isPipelineRunning ? { 'disable-pipeline': 'Disable pipeline' } : { 'enable-pipeline': 'Enable pipeline' }),
        ...PROJECT_ACTIONS,
    };

    const handleAction = (key: Key) => {
        const mutationParams = { params: { path: { project_id: projectId } } };

        switch (key) {
            case 'enable-pipeline':
                enablePipelineMutation.mutate(mutationParams, {
                    onSuccess: () => {
                        toast({ type: 'success', message: 'Pipeline enabled successfully' });
                    },
                });
                break;
            case 'disable-pipeline':
                disablePipelineMutation.mutate(mutationParams, {
                    onSuccess: () => {
                        toast({ type: 'success', message: 'Pipeline disabled successfully' });
                    },
                });
                break;
            case 'rename':
                callbacks.onRename();
                break;
            case 'delete':
                callbacks.onDelete();
                break;
            default:
                break;
        }
    };

    return {
        menuActions,
        handleAction,
    };
};
