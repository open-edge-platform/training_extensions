// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../api/client';

export const usePipeline = () => {
    const projectId = useProjectIdentifier();

    return $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });
};

const POLLING_INTERVAL = 5000;
export const usePipelineMetrics = () => {
    const projectId = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/pipeline/metrics',
        {
            params: { path: { project_id: projectId } },
        },
        {
            refetchInterval: (query) => (query.state.status === 'success' ? POLLING_INTERVAL : false),
            retry: false,
        }
    );
};

export const usePatchPipeline = () => {
    return $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: { invalidateQueries: [['get', '/api/projects/{project_id}/pipeline']] },
    });
};

export const useEnablePipeline = () => {
    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:enable', {
        meta: { invalidateQueries: [['get', '/api/projects/{project_id}/pipeline']] },
    });
};

export const useDisablePipeline = () => {
    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:disable', {
        meta: { invalidateQueries: [['get', '/api/projects/{project_id}/pipeline']] },
    });
};

export const useConnectSourceToPipeline = () => {
    const project_id = useProjectIdentifier();
    const pipeline = usePatchPipeline();

    return (source_id: string) => pipeline.mutateAsync({ params: { path: { project_id } }, body: { source_id } });
};
