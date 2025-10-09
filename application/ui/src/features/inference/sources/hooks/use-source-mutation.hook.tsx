// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { omit } from 'lodash-es';
import { queryClient } from 'src/providers';
import { v4 as uuid } from 'uuid';

import { $api } from '../../../../api/client';
import { SourceConfig } from '../util';

export const useSourceMutation = (isNewSource: boolean) => {
    const addSource = $api.useMutation('post', '/api/sources');
    const updateSource = $api.useMutation('patch', '/api/sources/{source_id}');

    return async (body: SourceConfig) => {
        if (isNewSource) {
            const sourcePayload = {
                ...body,
                id: uuid(),
            };

            const response = await addSource.mutateAsync({ body: sourcePayload });

            return String(response.id);
        }

        const response = await updateSource.mutateAsync({
            params: { path: { source_id: String(body.id) } },
            body: omit(body, 'source_type'),
        });

        queryClient.invalidateQueries({ queryKey: ['get', `/api/sources`] });
        queryClient.invalidateQueries({ queryKey: ['get', `/api/sources/{source_id}`] });

        return String(response.id);
    };
};
