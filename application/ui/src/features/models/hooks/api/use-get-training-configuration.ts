// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import { TrainingConfiguration } from '../../configuration.interface';

const useGetTrainingConfigurationQuery = (modelArchitectureId: string | null) => {
    const projectId = useProjectIdentifier();

    return $api.useQuery('get', '/api/projects/{project_id}/training_configuration', {
        params: {
            path: {
                project_id: projectId,
            },
            query: {
                model_architecture_id: modelArchitectureId,
            },
        },
    });
};

// NOTE: Currently backend does return generic dictionary, so we need to cast it to TrainingConfiguration
export const useGetTrainingConfiguration = (modelArchitectureId: string | null) => {
    const { data } = useGetTrainingConfigurationQuery(modelArchitectureId);

    const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>(
        data as TrainingConfiguration | undefined
    );

    useEffect(() => {
        if (data !== undefined) {
            setTrainingConfiguration(data as unknown as TrainingConfiguration);
        }
    }, [data]);

    return [trainingConfiguration, setTrainingConfiguration, data as TrainingConfiguration | undefined] as const;
};
