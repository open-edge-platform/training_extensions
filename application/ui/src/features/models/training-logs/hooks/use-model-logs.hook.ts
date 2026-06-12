// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMutation, useQuery } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { fetchClient } from '../../../../api/client';
import { toast } from '../../../../components/toast/toast.component';
import { getQueryKey } from '../../../../query-client/query-client';
import { downloadFile } from '../../../../shared/util';
import { type LogEntry } from '../log-types';
import { parseLogLine } from '../log-utils';

const fetchModelLogs = async (projectId: string, modelId: string): Promise<LogEntry[]> => {
    const { data, error, response } = await fetchClient.GET('/api/projects/{project_id}/models/{model_id}/logs', {
        params: { path: { project_id: projectId, model_id: modelId } },
        parseAs: 'text',
    });

    if (error) {
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

const downloadModelLogsFile = async (projectId: string, modelId: string) => {
    const { data, error, response } = await fetchClient.GET('/api/projects/{project_id}/models/{model_id}/logs', {
        params: {
            path: { project_id: projectId, model_id: modelId },
            header: { accept: 'text/plain' },
        },
        parseAs: 'blob',
    });

    if (error || !data) {
        throw new Error(`Failed to download model logs: ${response.status} ${response.statusText}`);
    }

    const url = URL.createObjectURL(data);
    downloadFile(url, `training-logs-${modelId}.log`, 'Training logs download started');
};

export const useDownloadModelLogs = (modelId: string) => {
    const projectId = useProjectIdentifier();

    const mutation = useMutation({
        mutationFn: async () => {
            if (!modelId) {
                throw new Error('Model id is required to download logs');
            }

            await downloadModelLogsFile(projectId, modelId);
        },
        onSuccess: () => {
            toast({ type: 'success', message: 'Training logs downloaded successfully' });
        },
        onError: () => {
            toast({ type: 'error', message: 'Failed to download training logs' });
        },
    });

    return {
        downloadModelLogs: () => mutation.mutate(),
        isDownloading: mutation.isPending,
    };
};
