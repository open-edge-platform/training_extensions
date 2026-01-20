// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import { DeviceType } from '../../../../constants/shared-types';

export const useTrainModelMutation = () => {
    const trainModelMutation = $api.useMutation('post', '/api/jobs', {
        meta: {
            invalidateQueries: [['get', '/api/jobs']],
        },
    });
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
