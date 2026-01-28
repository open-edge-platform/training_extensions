// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../api/client';
import type { DeviceType, ModelFormat } from '../../constants/shared-types';
import { useSubmitJob } from './jobs.hook';
import { usePipeline } from './pipeline.hook';
import { useProject } from './project.hook';

export const useDeleteModel = () => {
    return $api.useMutation('delete', '/api/projects/{project_id}/models/{model_id}', {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/models']],
        },
    });
};

export const useDownloadModel = (modelId: string) => {
    const projectId = useProjectIdentifier();

    const mutation = $api.useMutation('get', '/api/projects/{project_id}/models/{model_id}/binary', {
        onSuccess: (data, variables) => {
            const blob = data as Blob;
            const url = URL.createObjectURL(blob);
            const format = variables.params.query?.format;

            const link = document.createElement('a');
            link.href = url;
            link.download = format ? `model-${modelId}-${format}.zip` : `model-${modelId}.zip`;
            link.click();

            URL.revokeObjectURL(url);

            toast({ type: 'success', message: 'Model downloaded successfully' });
        },
        onError: () => {
            toast({ type: 'error', message: 'Failed to download model' });
        },
    });

    const downloadModel = (format?: ModelFormat) => {
        mutation.mutate({
            params: {
                path: { project_id: projectId, model_id: modelId },
                query: format ? { format } : undefined,
            },
            parseAs: 'blob',
        });
    };

    return {
        downloadModel,
        isDownloading: mutation.isPending,
        error: mutation.error,
    };
};

export const useGetActiveModelArchitectureId = () => {
    const pipeline = usePipeline();

    return pipeline.data.model?.architecture;
};

export const useGetDatasetRevisions = () => {
    return {
        data: [
            { id: '1', name: 'Dataset 1' },
            { id: '2', name: 'Dataset 2' },
            { id: '3', name: 'Dataset 3' },
            { id: '4', name: 'Dataset 4' },
        ],
    };
};

export const useGetTaskModelArchitectures = () => {
    const { data: projectData } = useProject();

    return $api.useSuspenseQuery('get', '/api/model_architectures', {
        params: {
            query: {
                task: projectData.task.task_type,
            },
        },
    });
};

export const useGetModel = (modelId: string | null | undefined) => {
    const projectId = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/models/{model_id}',
        { params: { path: { project_id: projectId, model_id: String(modelId) } } },
        { enabled: Boolean(modelId) }
    );
};

export const useGetModels = () => {
    const projectId = useProjectIdentifier();

    return $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id: projectId } },
    });
};

export const useGetTrainingDevices = () => {
    return $api.useSuspenseQuery('get', '/api/system/devices/training');
};

export const useRenameModel = () => {
    return $api.useMutation('patch', '/api/projects/{project_id}/models/{model_id}', {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/models']],
        },
    });
};

export const useTrainModelMutation = () => {
    const trainModelMutation = useSubmitJob();
    const projectIdentifier = useProjectIdentifier();

    const trainModel = (
        {
            device,
            modelArchitectureId,
        }: {
            device: DeviceType;
            modelArchitectureId: string;
            datasetRevisionId: string;
        },
        onSuccess?: () => void
    ) => {
        trainModelMutation.mutate(
            {
                body: {
                    job_type: 'train',
                    project_id: projectIdentifier,
                    parameters: {
                        device,
                        model_architecture_id: modelArchitectureId,
                        // TODO: uncomment once supported by backend
                        // dataset_revision_id: datasetRevisionId,
                    },
                },
            },
            {
                onSuccess,
            }
        );
    };

    return { mutate: trainModel, isPending: trainModelMutation.isPending };
};
