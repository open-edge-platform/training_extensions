// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, toast } from '@geti/ui';
import { useDisablePipeline, usePipeline } from 'hooks/api/pipeline.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

const useTogglePipeline = () => {
    const projectId = useProjectIdentifier();

    const pipelineQuery = usePipeline();
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

    return {
        isPipelineEnabled,
        deactivatePipeline: () => {
            disablePipelineMutation.mutate(mutationParams, mutationOptions);
        },
    };
};

export const DisablePipelineButton = () => {
    const { deactivatePipeline, isPipelineEnabled } = useTogglePipeline();

    return (
        <Button isDisabled={!isPipelineEnabled} onPress={deactivatePipeline}>
            Disable Pipeline
        </Button>
    );
};
