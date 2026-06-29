// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePrefetchQuery, useSuspenseQuery } from '@tanstack/react-query';

import { $api } from '../../../../api/client';

const sourcesQueryOptions = () => {
    return $api.queryOptions('get', '/api/sources');
};

export const usePrefetchSourcesQuery = () => {
    usePrefetchQuery(sourcesQueryOptions());
};

export const useSourcesQuery = () => {
    return useSuspenseQuery(sourcesQueryOptions());
};
