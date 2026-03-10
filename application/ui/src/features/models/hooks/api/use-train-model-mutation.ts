// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSubmitJob } from 'hooks/api/jobs/jobs.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import type { DeviceType } from '../../../../constants/shared-types';

export const useTrainModelMutation = () => {
    const trainModelMutation = useSubmitJob();
    const projectIdentifier = useProjectIdentifier();

    const trainModel = (
        {
            device,
            modelArchitectureId,
            datasetRevisionId,
            parentModelRevisionId,
        }: {
            device: DeviceType;
            modelArchitectureId: string;
            datasetRevisionId: string | null;
            parentModelRevisionId: string | null;
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
                        dataset_revision_id: datasetRevisionId,
                        parent_model_revision_id: parentModelRevisionId,
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
