// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';
import { omit } from 'lodash-es';

import { $api } from '../../../../api/client';
import { getQueryKey } from '../../../../query-client/query-client';
import { testSinkQueryOptions } from '../api/use-test-sink';
import { SinkConfig } from '../utils';

const useUpdateSink = () => {
    const queryClient = useQueryClient();

    return $api.useMutation('patch', '/api/sinks/{sink_id}', {
        meta: {
            invalidateQueries: [['get', '/api/sinks']],
        },
        onSuccess: (
            _,
            {
                params: {
                    path: { sink_id },
                },
            }
        ) => {
            return queryClient.invalidateQueries({
                queryKey: getQueryKey(['get', '/api/sinks/{sink_id}', { params: { path: { sink_id } } }]),
            });
        },
    });
};

export const useSinkMutation = (isNewSink: boolean) => {
    const queryClient = useQueryClient();

    const addSink = $api.useMutation('post', '/api/sinks', {
        meta: {
            invalidateQueries: [['get', '/api/sinks']],
        },
    });

    const updateSink = useUpdateSink();

    return async (body: SinkConfig) => {
        let sinkId: string;

        if (isNewSink) {
            const response = await addSink.mutateAsync({ body: omit(body, 'id') as SinkConfig });

            sinkId = String(response.id);
        } else {
            const response = await updateSink.mutateAsync({
                params: { path: { sink_id: String(body.id) } },
                body: omit(body, 'sink_type'),
            });

            sinkId = String(response.id);
        }

        void queryClient.fetchQuery({
            ...testSinkQueryOptions(sinkId),
            staleTime: 0,
        });

        return sinkId;
    };
};
