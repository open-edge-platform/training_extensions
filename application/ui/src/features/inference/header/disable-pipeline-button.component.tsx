// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, toast } from '@geti/ui';
import { useDisablePipeline, usePipeline } from 'hooks/api/pipeline.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

export const DisablePipelineButton = () => {
    const projectId = useProjectIdentifier();

    const pipelineQuery = usePipeline();
    const disablePipelineMutation = useDisablePipeline();

    const isPipelineEnabled = pipelineQuery.data?.status === 'running';

    const mutationParams = { params: { path: { project_id: projectId } } };
    const mutationOptions = {
        onSuccess: () => {
            toast({
                type: 'success',
                message: 'Pipeline disabled successfully',
            });
        },
        onError: () => {
            toast({
                type: 'error',
                message: 'Failed to disable pipeline',
            });
        },
    };

    return (
        <Button
            isDisabled={!isPipelineEnabled}
            onPress={() => disablePipelineMutation.mutate(mutationParams, mutationOptions)}
        >
            Disable Pipeline
        </Button>
    );
};
