// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from 'react';

import { TrainingConfiguration } from '../../../../constants/shared-types';
import {
    useGetModelArchitectureTrainingConfiguration,
    useGetModelTrainingConfiguration,
} from '../../hooks/api/use-get-model-training-configuration.hook';

type useTrainingConfigurationProps = {
    modelArchitectureId: string | null;
    modelRevisionId: string | null;
};

export const useTrainingConfiguration = ({ modelArchitectureId, modelRevisionId }: useTrainingConfigurationProps) => {
    const modelArchitectureTrainingConfigurationQuery = useGetModelArchitectureTrainingConfiguration({
        modelArchitectureId,
        modelRevisionId,
    });

    const modelTrainingConfigurationQuery = useGetModelTrainingConfiguration(modelRevisionId);

    useEffect(() => {
        if (modelTrainingConfigurationQuery.isSuccess) {
            setTrainingConfiguration(modelTrainingConfigurationQuery.data);
        }
    }, [modelTrainingConfigurationQuery.data, modelTrainingConfigurationQuery.isSuccess]);

    useEffect(() => {
        if (modelArchitectureTrainingConfigurationQuery.isSuccess) {
            setTrainingConfiguration(modelArchitectureTrainingConfigurationQuery.data);
        }
    }, [modelArchitectureTrainingConfigurationQuery.data, modelArchitectureTrainingConfigurationQuery.isSuccess]);

    const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>(undefined);

    return [
        trainingConfiguration,
        setTrainingConfiguration,
        modelRevisionId === null
            ? modelArchitectureTrainingConfigurationQuery.data
            : modelTrainingConfigurationQuery.data,
    ] as const;
};
