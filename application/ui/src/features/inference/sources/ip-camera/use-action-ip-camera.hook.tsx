// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { toast } from '@geti/ui';

import { $api } from '../../../../api/client';
import { useProjectIdentifier } from '../../../../hooks/use-project-identifier.hook';
import { useSourceMutation } from '../hooks/use-source-mutation.hook';
import { IPCameraSourceConfig } from '../util';

const iniConfig: IPCameraSourceConfig = {
    name: '',
    source_type: 'ip_camera',
    stream_url: '',
    auth_required: false,
};

export const useActionImageFolder = (config = iniConfig, isNewSource = false) => {
    const projectId = useProjectIdentifier();
    const pipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline');
    const addOrUpdateSource = useSourceMutation(isNewSource);

    return useActionState<IPCameraSourceConfig, FormData>(async (_prevState, formData) => {
        const body = {
            id: String(formData.get('id')),
            name: formData.get('name'),
            source_type: 'ip_camera',
            stream_url: formData.get('stream_url'),
            auth_required: formData.get('auth_required') === 'on' ? true : false,
        } as unknown as IPCameraSourceConfig;

        try {
            const source_id = await addOrUpdateSource(body);

            await pipeline.mutateAsync({
                params: { path: { project_id: projectId } },
                body: { source_id },
            });

            toast({
                type: 'success',
                message: `Image folder configuration ${isNewSource ? 'created' : 'updated'} successfully.`,
            });

            return { ...body, id: source_id };
        } catch (error: unknown) {
            const details = (error as { detail?: string })?.detail;

            toast({
                type: 'error',
                message: `Failed to save source configuration, ${details ?? 'please try again'}`,
            });
        }

        return body;
    }, config);
};
