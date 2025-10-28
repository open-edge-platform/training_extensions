// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { Switch, toast } from '@geti/ui';
import { useDisablePipeline, useEnablePipeline, usePipeline } from 'hooks/api/pipeline.hook';
import { useIsPipelineConfigured } from 'hooks/use-is-pipeline-configured.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { throttle } from 'lodash-es';

const DELAY = 2000;

const useTogglePipeline = () => {
    const projectId = useProjectIdentifier();

    const pipelineQuery = usePipeline();
    const canEditPipeline = useIsPipelineConfigured(pipelineQuery.data);

    const enablePipelineMutation = useEnablePipeline();
    const disablePipelineMutation = useDisablePipeline();

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
        isDisabled: enablePipelineMutation.isPending || disablePipelineMutation.isPending || !canEditPipeline,
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
