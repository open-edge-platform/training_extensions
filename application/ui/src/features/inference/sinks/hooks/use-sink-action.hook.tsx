// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { toast } from '@geti/ui';

import { $api } from '../../../../api/client';
import { useProjectIdentifier } from '../../../../hooks/use-project-identifier.hook';
import { SinkConfig } from '../utils';
import { useSinkMutation } from './use-sink-mutation.hook';

interface useSinkActionProps<T> {
    config: Awaited<T>;
    isNewSink: boolean;
    bodyFormatter: (formData: FormData) => T;
}

export const useSinkAction = <T extends SinkConfig>({ config, isNewSink, bodyFormatter }: useSinkActionProps<T>) => {
    const projectId = useProjectIdentifier();
    const pipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline');
    const addOrUpdateSink = useSinkMutation(isNewSink);

    return useActionState<T, FormData>(async (_prevState: T, formData: FormData) => {
        const body = bodyFormatter(formData);

        try {
            const sink_id = await addOrUpdateSink(body);

            await pipeline.mutateAsync({
                params: { path: { project_id: projectId } },
                body: { sink_id },
            });

            toast({
                type: 'success',
                message: `Sink configuration ${isNewSink ? 'created' : 'updated'} successfully.`,
            });

            return { ...body, id: sink_id };
        } catch (error: unknown) {
            const details = (error as { detail?: string })?.detail;

            toast({
                type: 'error',
                message: `Failed to save sink configuration, ${details ?? 'please try again'}`,
            });
        }

        return body;
    }, config);
};
