// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { Switch, toast } from '@geti/ui';
import { throttle } from 'lodash-es';
import { $api } from 'src/api/client';
import { useProjectIdentifier } from 'src/hooks/use-project-identifier.hook';

const DELAY = 2000;

const useTogglePipeline = () => {
    const projectId = useProjectIdentifier();

    const pipelineQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });

    const enablePipelineMutation = $api.useMutation('post', '/api/projects/{project_id}/pipeline:enable');
    const disablePipelineMutation = $api.useMutation('post', '/api/projects/{project_id}/pipeline:disable');

    const isPipelineEnabled = pipelineQuery.data?.status === 'running';

    const mutationParams = { params: { path: { project_id: projectId } } };
    const mutationOptions = {
        onSuccess: () => {
            toast({
                type: 'success',
                message: isPipelineEnabled ? 'Pipeline disabled successfully' : 'Pipeline enabled successfully',
            });
        },
        onError: () => {
            toast({
                type: 'error',
                message: isPipelineEnabled ? 'Failed to disable pipeline' : 'Failed to enable pipeline',
            });
        },
    };

    const activatePipelineRef = useRef<(() => void) | null>(null);
    const deactivatePipelineRef = useRef<(() => void) | null>(null);

    if (!activatePipelineRef.current) {
        activatePipelineRef.current = throttle(() => {
            enablePipelineMutation.mutate(mutationParams, mutationOptions);
        }, DELAY);
    }

    if (!deactivatePipelineRef.current) {
        deactivatePipelineRef.current = throttle(() => {
            disablePipelineMutation.mutate(mutationParams, mutationOptions);
        }, DELAY);
    }

    return {
        isPipelineEnabled,
        isDisabled: enablePipelineMutation.isPending || disablePipelineMutation.isPending,
        activatePipeline: activatePipelineRef.current,
        deactivatePipeline: deactivatePipelineRef.current,
    };
};

export const PipelineSwitch = () => {
    const { activatePipeline, deactivatePipeline, isPipelineEnabled, isDisabled } = useTogglePipeline();

    return (
        <Switch
            isSelected={isPipelineEnabled}
            isDisabled={isDisabled}
            onChange={(isSelected) => {
                if (isSelected) {
                    activatePipeline();
                } else {
                    deactivatePipeline();
                }
            }}
        >
            {`${isPipelineEnabled ? 'Disable' : 'Enable'} Pipeline`}
        </Switch>
    );
};
