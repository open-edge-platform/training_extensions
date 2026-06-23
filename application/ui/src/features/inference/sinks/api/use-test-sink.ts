// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';

import { $api } from '../../../../api/client';

export const testSinkQueryOptions = (sinkId: string) =>
    $api.queryOptions(
        'post',
        '/api/sinks/{sink_id}:test',
        { params: { path: { sink_id: sinkId } } },
        { enabled: false, staleTime: Infinity, gcTime: Infinity }
    );

export const useTestSink = (sinkId: string) => {
    return useQuery(testSinkQueryOptions(sinkId));
};
