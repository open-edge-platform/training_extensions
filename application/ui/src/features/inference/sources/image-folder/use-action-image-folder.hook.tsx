// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { toast } from '@geti/ui';
import { omit } from 'lodash-es';

import { $api } from '../../../../api/client';
import { useProjectIdentifier } from '../../../../hooks/use-project-identifier.hook';
import { ImagesFolderSourceConfig } from '../util';

const iniConfig: ImagesFolderSourceConfig = {
    name: '',
    source_type: 'images_folder',
    images_folder_path: '',
    ignore_existing_images: false,
};

const useMutationSource = (isNewSource: boolean) => {
    const addSource = $api.useMutation('post', '/api/sources');
    const updateSource = $api.useMutation('patch', '/api/sources/{source_id}');

    return async (body: ImagesFolderSourceConfig) => {
        if (isNewSource) {
            const response = await addSource.mutateAsync({ body: omit(body, 'id') });

            return String(response.id);
        }

        const response = await updateSource.mutateAsync({
            params: { path: { source_id: String(body.id) } },
            body: omit(body, 'source_type'),
        });

        return String(response.id);
    };
};

export const useActionImageFolder = (config = iniConfig, isNewSource = false) => {
    const projectId = useProjectIdentifier();
    const pipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline');
    const addOrUpdateSource = useMutationSource(isNewSource);

    return useActionState<ImagesFolderSourceConfig, FormData>(async (_prevState, formData) => {
        const body = {
            id: String(formData.get('id')),
            name: formData.get('name'),
            source_type: 'images_folder',
            images_folder_path: formData.get('images_folder_path'),
            ignore_existing_images: formData.get('ignore_existing_images') === 'on' ? true : false,
        } as ImagesFolderSourceConfig;

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
