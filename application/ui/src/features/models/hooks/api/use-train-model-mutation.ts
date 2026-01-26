// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSubmitJob } from 'hooks/api/jobs.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { DeviceType } from '../../../../constants/shared-types';

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
