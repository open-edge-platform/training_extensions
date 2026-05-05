// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef } from 'react';

import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../api/client';
import { Job } from '../../../constants/shared-types';
import { getQueryKey } from '../../../query-client/query-client';
import { useSSE } from '../../use-sse.hook';
import { isQuantizeJob, isTrainJob } from '../util';

const TERMINAL_STATUSES: string[] = ['DONE', 'FAILED', 'CANCELLED'];

export const useStreamJobStatus = (jobId: string | undefined) => {
    const queryClient = useQueryClient();
    const projectId = useProjectIdentifier();
    const modelIdRef = useRef<string | null>(null);

    const { close } = useSSE<Job>(jobId ? `/api/jobs/${jobId}/status` : undefined, {
        onMessage: (updatedJob) => {
            if (isQuantizeJob(updatedJob)) {
                modelIdRef.current = updatedJob.metadata.model.id;
            }
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

            modelIdRef.current !== null &&
                queryClient.invalidateQueries({
                    queryKey: getQueryKey([
                        'get',
                        '/api/projects/{project_id}/models/{model_id}',
                        {
                            params: {
                                path: {
                                    project_id: projectId,
                                    model_id: modelIdRef.current,
                                },
                            },
                        },
                    ]),
                });
            modelIdRef.current = null;
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

    const activeRunningJobs = activeJobs.data?.filter((job) => {
        const isActive = job.status === 'RUNNING' || job.status === 'PENDING';

        if (isActive && (isTrainJob(job) || isQuantizeJob(job))) {
            return job.metadata.project.id === projectId;
        }

        return false;
    });

    return activeRunningJobs;
};

export const useCancelJob = () => {
    return $api.useMutation('post', '/api/jobs/{job_id}:cancel', {
        meta: {
            invalidateQueries: [['get', '/api/jobs']],
        },
    });
};
