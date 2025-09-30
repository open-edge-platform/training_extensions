// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Switch, toast } from '@geti/ui';
import { $api } from 'src/api/client';
import { useProjectIdentifier } from 'src/hooks/use-project-identifier.hook';

export const PipelineSwitch = () => {
    const projectId = useProjectIdentifier();

    const pipelineQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });

    const enablePipelineMutation = $api.useMutation('post', '/api/projects/{project_id}/pipeline:enable');
    const disablePipelineMutation = $api.useMutation('post', '/api/projects/{project_id}/pipeline:disable');

    const isPipelineEnabled = pipelineQuery.data?.status === 'running';

    const mutationParams = { params: { path: { project_id: projectId } } };
    const enableMutationOptions = {
        onSuccess: () => {
            toast({ type: 'success', message: 'Pipeline enabled successfully' });
        },
        onError: () => {
            toast({ type: 'error', message: 'Failed to enable pipeline' });
        },
    };
    const disableMutationOptions = {
        onSuccess: () => {
            toast({ type: 'success', message: 'Pipeline disabled successfully' });
        },
        onError: () => {
            toast({ type: 'error', message: 'Failed to disable pipeline' });
        },
    };

    return (
        <Switch
            isSelected={isPipelineEnabled}
            isDisabled={enablePipelineMutation.isPending || disablePipelineMutation.isPending}
            onChange={(isSelected) => {
                if (isSelected) {
                    enablePipelineMutation.mutate(mutationParams, enableMutationOptions);
                } else {
                    disablePipelineMutation.mutate(mutationParams, disableMutationOptions);
                }
            }}
        >
            {`${isPipelineEnabled ? 'Disable' : 'Enable'} Pipeline`}
        </Switch>
    );
};
