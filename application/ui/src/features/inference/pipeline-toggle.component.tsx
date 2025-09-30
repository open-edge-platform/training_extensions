// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { Switch, toast } from '@geti/ui';
import { $api } from 'src/api/client';
import { useProjectIdentifier } from 'src/hooks/use-project-identifier.hook';

export const PipelineSwitch = () => {
    const projectId = useProjectIdentifier();

    const enablePipelineMutation = $api.useMutation('post', '/api/projects/{project_id}/pipeline:enable');
    const disablePipelineMutation = $api.useMutation('post', '/api/projects/{project_id}/pipeline:disable');

    const isPipelineEnabled = useRef(false);

    const mutationParams = { params: { path: { project_id: projectId } } };
    const mutationOptions = {
        onSuccess: () => {
            toast({ type: 'success', message: 'Pipeline enabled successfully' });
        },
        onError: () => {
            toast({ type: 'error', message: 'Failed to enable pipeline' });
        },
    };

    return (
        <Switch
            onChange={(isSelected) => {
                if (isSelected) {
                    isPipelineEnabled.current = true;
                    enablePipelineMutation.mutate(mutationParams, mutationOptions);
                } else {
                    isPipelineEnabled.current = false;
                    disablePipelineMutation.mutate(mutationParams, mutationOptions);
                }
            }}
        >
            {`${isPipelineEnabled.current ? 'Disable' : 'Enable'} Pipeline`}
        </Switch>
    );
};
