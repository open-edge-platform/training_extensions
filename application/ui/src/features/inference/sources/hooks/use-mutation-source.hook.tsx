// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { omit } from 'lodash-es';

import { $api } from '../../../../api/client';
import { SourceConfig } from './../util';

export const useMutationSource = (isNewSource: boolean) => {
    const addSource = $api.useMutation('post', '/api/sources');
    const updateSource = $api.useMutation('patch', '/api/sources/{source_id}');

    return async (body: SourceConfig) => {
        if (isNewSource) {
            const response = await addSource.mutateAsync({ body: omit(body, 'id') as SourceConfig });

            return String(response.id);
        }

        const response = await updateSource.mutateAsync({
            params: { path: { source_id: String(body.id) } },
            body: omit(body, 'source_type'),
        });

        return String(response.id);
    };
};
