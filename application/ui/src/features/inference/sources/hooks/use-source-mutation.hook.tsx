// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';
import { omit } from 'lodash-es';
import { v4 as uuid } from 'uuid';

import { $api } from '../../../../api/client';
import type { SourceConfigPayload } from '../../../../constants/shared-types';
import { getQueryKey } from '../../../../query-client/query-client';
import { testSourceQueryOptions } from '../api/use-test-source';

const useUpdateSource = () => {
    const queryClient = useQueryClient();

    return $api.useMutation('patch', '/api/sources/{source_id}', {
        meta: {
            invalidateQueries: [['get', '/api/sources']],
        },
        onSuccess: (
            _,
            {
                params: {
                    path: { source_id },
                },
            }
        ) => {
            return queryClient.invalidateQueries({
                queryKey: getQueryKey(['get', '/api/sources/{source_id}', { params: { path: { source_id } } }]),
            });
        },
    });
};

export const useSourceMutation = (isNewSource: boolean) => {
    const queryClient = useQueryClient();

    const addSource = $api.useMutation('post', '/api/sources', {
        meta: {
            invalidateQueries: [['get', '/api/sources']],
        },
    });

    const updateSource = useUpdateSource();

    return async (body: SourceConfigPayload) => {
        let sourceId: string;

        if (isNewSource) {
            const sourcePayload = {
                ...body,
                id: uuid(),
            };

            const response = await addSource.mutateAsync({ body: sourcePayload });

            sourceId = String(response.id);
        } else {
            const response = await updateSource.mutateAsync({
                params: { path: { source_id: String(body.id) } },
                body: omit(body, 'source_type'),
            });

            sourceId = String(response.id);
        }

        void queryClient.fetchQuery({
            ...testSourceQueryOptions(sourceId),
            staleTime: 0,
        });

        return sourceId;
    };
};
