// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { omit } from 'lodash-es';

import { $api } from '../../../../api/client';
import { SinkConfig } from '../utils';

export const useSinkMutation = (isNewSink: boolean) => {
    const addSink = $api.useMutation('post', '/api/sinks', {
        meta: {
            invalidateQueries: [['get', '/api/sinks']],
        },
    });
    const updateSink = $api.useMutation('patch', '/api/sinks/{sink_id}', {
        meta: {
            invalidateQueries: [
                ['get', '/api/sinks'],
                ['get', '/api/sinks/{sink_id}'],
            ],
        },
    });

    return async (body: SinkConfig) => {
        if (isNewSink) {
            const response = await addSink.mutateAsync({ body: omit(body, 'id') as SinkConfig });

            return String(response.id);
        }

        const response = await updateSink.mutateAsync({
            params: { path: { sink_id: String(body.id) } },
            body: omit(body, 'sink_type'),
        });

        return String(response.id);
    };
};
