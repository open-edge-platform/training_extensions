// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useSSE } from '../../../../hooks/use-sse.hook';
import { type LogEntry } from '../log-types';

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';

type UseStreamJobLogsReturn = {
    logs: LogEntry[];
    connectionStatus: ConnectionStatus;
};

export const useStreamJobLogs = (jobId: string | undefined): UseStreamJobLogsReturn => {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');

    useSSE<LogEntry>(jobId ? `/api/jobs/${jobId}/logs` : undefined, {
        retry: true,
        onMessage: (entry) => {
            setLogs((prev) => [...prev, entry]);
        },
        onOpen: () => {
            setConnectionStatus('connected');
        },
        onError: () => {
            setConnectionStatus('connecting');
        },
        onClose: () => {
            setConnectionStatus('disconnected');
        },
    });

    return { logs, connectionStatus };
};
