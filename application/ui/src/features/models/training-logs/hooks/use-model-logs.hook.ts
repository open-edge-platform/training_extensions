// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { fetchClient } from '../../../../api/client';
import { getQueryKey } from '../../../../query-client/query-client';
import { type LogEntry } from '../log-types';
import { parseLogLine } from '../log-utils';

const fetchModelLogs = async (projectId: string, modelId: string): Promise<LogEntry[]> => {
    const { data, response } = await fetchClient.GET('/api/projects/{project_id}/models/{model_id}/logs', {
        params: { path: { project_id: projectId, model_id: modelId } },
        parseAs: 'text',
    });

    if (!response.ok) {
        throw new Error(`Failed to fetch model logs: ${response.status} ${response.statusText}`);
    }

    const text = data ?? '';

    return text
        .split('\n')
        .filter((line) => line.trim())
        .map((line) => parseLogLine(line))
        .filter((entry): entry is LogEntry => entry !== null);
};

export const useModelLogs = (modelId: string | undefined) => {
    const projectId = useProjectIdentifier();

    return useQuery({
        queryKey: getQueryKey([
            'get',
            '/api/projects/{project_id}/models/{model_id}/logs',
            { params: { path: { project_id: projectId, model_id: modelId! } } },
        ]),
        queryFn: () => fetchModelLogs(projectId, modelId!),
        enabled: !!modelId,
        staleTime: Infinity, // Completed/failed model logs don't change
    });
};
