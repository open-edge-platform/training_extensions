// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../api/client';

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
    const activeJobs = $api.useQuery('get', '/api/jobs', undefined, {
        refetchInterval: (query) => {
            const hasActiveJob = query.state.data?.some(
                (job) => job.status === 'RUNNING' || job.status === 'PENDING' || job.status === 'CANCELLING'
            );

            return hasActiveJob ? 5000 : false;
        },
    });

    const activeTrainingJob = activeJobs.data?.find(
        (job) => job.metadata.project.id === projectId && job.status === 'RUNNING' && job.job_type === 'train'
    );

    return activeTrainingJob;
};

export const useCancelJob = () => {
    return $api.useMutation('post', '/api/jobs/{job_id}:cancel', {
        meta: {
            invalidateQueries: [['get', '/api/jobs']],
        },
    });
};
