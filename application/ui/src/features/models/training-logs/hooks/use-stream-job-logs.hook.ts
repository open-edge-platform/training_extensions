// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState } from 'react';

import { useSSE } from '../../../../hooks/use-sse.hook';
import { type LogEntry } from '../log-types';

type UseStreamJobLogsReturn = {
    logs: LogEntry[];
    isConnected: boolean;
    error: Error | null;
};

export const useStreamJobLogs = (jobId: string | undefined): UseStreamJobLogsReturn => {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<Error | null>(null);
    const logsRef = useRef<LogEntry[]>([]);

    useSSE<LogEntry>(jobId ? `/api/jobs/${jobId}/logs` : undefined, {
        onMessage: (entry) => {
            logsRef.current = [...logsRef.current, entry];
            setLogs(logsRef.current);
        },
        onOpen: () => {
            setIsConnected(true);
        },
        onError: () => {
            setIsConnected(false);
            setError(new Error('SSE connection lost'));
        },
        onClose: () => {
            setIsConnected(false);
        },
    });

    return { logs, isConnected, error };
};
