// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { AlertDialog, Button, DialogContainer, toast } from '@geti/ui';
import { useDisablePipeline, useEnablePipeline, usePipeline } from 'hooks/api/pipeline.hook';
import { useIsPipelineConfigured } from 'hooks/use-is-pipeline-configured.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

export const TogglePipelineButton = () => {
    const projectId = useProjectIdentifier();
    const [isEnableBlockedDialogOpen, setIsEnableBlockedDialogOpen] = useState(false);

    const pipelineQuery = usePipeline();
    const disablePipelineMutation = useDisablePipeline();
    const enablePipelineMutation = useEnablePipeline();

    const isPipelineEnabled = pipelineQuery.data?.status === 'running';
    const canEnablePipeline = useIsPipelineConfigured(pipelineQuery.data);

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
            if (!canEnablePipeline) {
                setIsEnableBlockedDialogOpen(true);
                return;
            }

            enablePipelineMutation.mutate(mutationParams, mutationOptions);
        }
    };

    return (
        <>
            <Button
                isPending={disablePipelineMutation.isPending || enablePipelineMutation.isPending}
                onPress={handleToggle}
            >
                {isPipelineEnabled ? 'Disable' : 'Enable'} Pipeline
            </Button>

            <DialogContainer onDismiss={() => setIsEnableBlockedDialogOpen(false)}>
                {isEnableBlockedDialogOpen && (
                    <AlertDialog
                        title={'Cannot enable pipeline'}
                        variant={'warning'}
                        primaryActionLabel={'Close'}
                        onPrimaryAction={() => setIsEnableBlockedDialogOpen(false)}
                    >
                        Make sure you selected a model, source, and sink before enabling the pipeline.
                    </AlertDialog>
                )}
            </DialogContainer>
        </>
    );
};
