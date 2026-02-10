// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from 'react';

import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../api/client';
import { connectSSE, SSEConnection } from '../../api/fetch-sse';
import type { Job } from '../../constants/shared-types';

const TERMINAL_STATUSES: string[] = ['DONE', 'FAILED', 'CANCELLED'];

/**
 * Subscribes to SSE status updates for a specific job and writes each update
 * into the React Query cache for `['get', '/api/jobs']`.
 *
 * The connection is automatically closed when:
 * - The job reaches a terminal status (DONE, FAILED, CANCELLED)
 * - The component unmounts
 * - The `jobId` changes
 */
export const useStreamJobStatus = (jobId: string | undefined) => {
    const queryClient = useQueryClient();
    const connectionRef = useRef<SSEConnection | null>(null);

    useEffect(() => {
        if (!jobId) {
            return;
        }

        connectionRef.current?.close();

        const connection = connectSSE<Job>(`/api/jobs/${jobId}/status`, {
            onMessage: (updatedJob) => {
                queryClient.setQueryData<Job[]>(['get', '/api/jobs'], (prevJobs) => {
                    if (!prevJobs) {
                        return [updatedJob];
                    }

                    return prevJobs.map((job) => (job.job_id === updatedJob.job_id ? updatedJob : job));
                });

                if (TERMINAL_STATUSES.includes(updatedJob.status)) {
                    connection.close();
                }
            },
            onClose: () => {
                connectionRef.current = null;
                queryClient.invalidateQueries({ queryKey: ['get', '/api/jobs'] });
                queryClient.invalidateQueries({ queryKey: ['get', '/api/projects/{project_id}/models'] });
            },
        });

        connectionRef.current = connection;

        return () => {
            connection.close();
        };
    }, [jobId, queryClient]);
};

export const useSubmitJob = () => {
    return $api.useMutation('post', '/api/jobs', {
        meta: {
            invalidateQueries: [['get', '/api/jobs']],
        },
    });
};

export const useListJobs = () => {
    return $api.useQuery('get', '/api/jobs');
};

export const useGetCurrentTrainingJob = () => {
    const projectId = useProjectIdentifier();
    const activeJobs = $api.useQuery('get', '/api/jobs');

    const activeTrainingJob = activeJobs.data?.find((job) => {
        const jobProjectId =
            'project' in job.metadata &&
            job.metadata.project &&
            'id' in job.metadata.project &&
            job.metadata.project.id;
        const isActive = job.status === 'RUNNING' || job.status === 'PENDING';
        return jobProjectId === projectId && isActive && job.job_type === 'train';
    });

    useStreamJobStatus(activeTrainingJob?.job_id);

    return activeTrainingJob;
};

export const useCancelJob = () => {
    return $api.useMutation('post', '/api/jobs/{job_id}:cancel', {
        meta: {
            invalidateQueries: [['get', '/api/jobs']],
        },
    });
};
