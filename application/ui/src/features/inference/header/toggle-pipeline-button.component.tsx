// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, toast } from '@geti/ui';
import { useDisablePipeline, useEnablePipeline, usePipeline } from 'hooks/api/pipeline.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

export const TogglePipelineButton = () => {
    const projectId = useProjectIdentifier();

    const pipelineQuery = usePipeline();
    const disablePipelineMutation = useDisablePipeline();
    const enablePipelineMutation = useEnablePipeline();

    const isPipelineEnabled = pipelineQuery.data?.status === 'running';

    const mutationParams = { params: { path: { project_id: projectId } } };

    const handleToggle = () => {
        const mutationOptions = {
            onSuccess: () => {
                toast({
                    type: 'success',
                    message: `Pipeline ${isPipelineEnabled ? 'disabled' : 'enabled'} successfully`,
                });
            },
        };

        if (isPipelineEnabled) {
            disablePipelineMutation.mutate(mutationParams, mutationOptions);
        } else {
            enablePipelineMutation.mutate(mutationParams, mutationOptions);
        }
    };

    return (
        <Button
            isPending={disablePipelineMutation.isPending || enablePipelineMutation.isPending}
            onPress={handleToggle}
        >
            {isPipelineEnabled ? 'Disable' : 'Enable'} Pipeline
        </Button>
    );
};
