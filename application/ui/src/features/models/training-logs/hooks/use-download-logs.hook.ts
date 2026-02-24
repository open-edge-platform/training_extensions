// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { fetchClient } from '../../../../api/client';

const sanitizeFilename = (name: string): string => {
    return name.replace(/[^a-zA-Z0-9_-]/g, '-').replace(/-+/g, '-');
};

const formatDate = (isoDate?: string): string => {
    const date = isoDate ? new Date(isoDate) : new Date();

    if (isNaN(date.getTime())) {
        return formatDate();
    }

    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');

    return `${year}-${month}-${day}`;
};

export const buildLogFilename = (modelName?: string, trainingDate?: string): string => {
    const parts = [modelName ? sanitizeFilename(modelName) : 'training', formatDate(trainingDate), 'logs'];

    return `${parts.join('-')}.txt`;
};

const downloadTextAsFile = (text: string, filename: string): void => {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');

    link.href = url;
    link.download = filename;
    link.hidden = true;
    link.click();

    URL.revokeObjectURL(url);
};

const fetchRawModelLogs = async (projectId: string, modelId: string): Promise<string> => {
    const { data, error, response } = await fetchClient.GET('/api/projects/{project_id}/models/{model_id}/logs', {
        params: {
            path: { project_id: projectId, model_id: modelId },
            header: { accept: 'text/plain' },
        },
        parseAs: 'text',
    });

    if (error) {
        throw new Error(`Failed to download logs: ${response.status} ${response.statusText}`);
    }

    return data ?? '';
};

type UseDownloadLogsProps = {
    modelId: string;
    modelName?: string;
    trainingDate?: string;
};

export const useDownloadLogs = ({ modelId, modelName, trainingDate }: UseDownloadLogsProps) => {
    const projectId = useProjectIdentifier();

    const downloadLogs = useCallback(async () => {
        const filename = buildLogFilename(modelName, trainingDate);
        const rawText = await fetchRawModelLogs(projectId, modelId);

        downloadTextAsFile(rawText, filename);
    }, [modelId, modelName, trainingDate, projectId]);

    return { downloadLogs };
};
