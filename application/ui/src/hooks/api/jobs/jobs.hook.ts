// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../api/client';
import { Job } from '../../../constants/shared-types';
import { getQueryKey } from '../../../query-client/query-client';
import { useSSE } from '../../use-sse.hook';

const TERMINAL_STATUSES: string[] = ['DONE', 'FAILED', 'CANCELLED'];

const useStreamJobStatus = (jobId: string | undefined) => {
    const queryClient = useQueryClient();
    const projectId = useProjectIdentifier();

    const { close } = useSSE<Job>(jobId ? `/api/jobs/${jobId}/status` : undefined, {
        onMessage: (updatedJob) => {
            // Update the job in the cache optimistically to reflect real-time progress
            queryClient.setQueryData<Job[]>(['get', '/api/jobs'], (prevJobs) => {
                if (!prevJobs) {
                    return [updatedJob];
                }

                return prevJobs.map((job) => (job.job_id === updatedJob.job_id ? updatedJob : job));
            });

            if (TERMINAL_STATUSES.includes(updatedJob.status)) {
                close();
            }
        },
        onClose: () => {
            queryClient.invalidateQueries({ queryKey: getQueryKey(['get', '/api/jobs']) });
            queryClient.invalidateQueries({
                queryKey: getQueryKey([
                    'get',
                    '/api/projects/{project_id}/models',
                    { params: { path: { project_id: projectId } } },
                ]),
            });
        },
    });
};

export const useSubmitJob = () => {
    return $api.useMutation('post', '/api/jobs', {
        meta: {
            invalidateQueries: [['get', '/api/jobs']],
        },
    });
};

const useListJobs = () => {
    return $api.useQuery('get', '/api/jobs');
};

export const useGetCurrentRunningJob = () => {
    const projectId = useProjectIdentifier();
    const activeJobs = useListJobs();

    const activeRunningJob = activeJobs.data?.find((job) => {
        const jobProjectId =
            'project' in job.metadata &&
            job.metadata.project &&
            'id' in job.metadata.project &&
            job.metadata.project.id;
        const isActive = job.status === 'RUNNING' || job.status === 'PENDING';

        return jobProjectId === projectId && isActive && (job.job_type === 'train' || job.job_type === 'quantize');
    });

    useStreamJobStatus(activeRunningJob?.job_id);

    return activeRunningJob;
};

export const useCancelJob = () => {
    return $api.useMutation('post', '/api/jobs/{job_id}:cancel', {
        meta: {
            invalidateQueries: [['get', '/api/jobs']],
        },
    });
};
