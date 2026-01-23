// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEqual } from 'lodash-es';

import { useTrainModelMutation } from '../hooks/api/use-train-model-mutation';
import { useUpdateTrainingConfiguration } from '../hooks/api/use-update-training-configuration';
import { useTrainModelState } from './train-model-provider.component';

export const useTrainModel = () => {
    const projectIdentifier = useProjectIdentifier();
    const {
        isAdvancedSettingsMode,
        selectedDatasetRevision,
        selectedModelArchitectureId,
        selectedTrainingDevice,
        trainingConfiguration,
        defaultTrainingConfiguration,
    } = useTrainModelState();

    const trainingConfigurationMutation = useUpdateTrainingConfiguration();
    const trainModelMutation = useTrainModelMutation();

    const trainModel = (onSuccess: () => void) => {
        // 1. If we are in basic mode, we can directly train the model, without updating the training configuration.
        // 2. If we are in advanced settings mode and training configuration is not changed, we can directly train the
        // model.
        // 3. If we are in advanced settings mode, we need to update the training configuration first.
        // 3.1. If the training configuration fails, we don't want to train the model.
        // 3.2. If the training configuration succeeds, we can train the model with the updated configuration.
        // 4. Train model is called.
        // 4.1. If train model fails, we revert the training configuration to the default one.
        // 4.2. If train model succeeds, we call the onSuccess callback if provided.
        if (selectedTrainingDevice === null || selectedDatasetRevision === null || selectedModelArchitectureId === null)
            return;

        if (!isAdvancedSettingsMode || isEqual(trainingConfiguration, defaultTrainingConfiguration)) {
            trainModelMutation.mutate(
                {
                    device: selectedTrainingDevice,
                    datasetRevisionId: selectedDatasetRevision,
                    modelArchitectureId: selectedModelArchitectureId,
                },
                { onSuccess }
            );

            return;
        }

        if (trainingConfiguration === undefined || defaultTrainingConfiguration === undefined) {
            return;
        }

        trainingConfigurationMutation.mutate(
            {
                params: {
                    path: {
                        project_id: projectIdentifier,
                    },
                    query: {
                        model_architecture_id: selectedModelArchitectureId,
                    },
                },
                // @ts-expect-error Body is not strongly typed in API yet
                body: trainingConfiguration,
            },
            {
                onSuccess: () => {
                    trainModelMutation.mutate(
                        {
                            device: selectedTrainingDevice,
                            datasetRevisionId: selectedDatasetRevision,
                            modelArchitectureId: selectedModelArchitectureId,
                        },
                        {
                            onSuccess,
                            onError: () => {
                                trainingConfigurationMutation.mutate({
                                    params: {
                                        path: {
                                            project_id: projectIdentifier,
                                        },
                                        query: {
                                            model_architecture_id: selectedModelArchitectureId,
                                        },
                                    },
                                    // @ts-expect-error Body is not strongly typed in API yet
                                    body: defaultTrainingConfiguration,
                                });
                            },
                        }
                    );
                },
            }
        );
    };

    return { trainModel, isPending: trainModelMutation.isPending || trainingConfigurationMutation.isPending };
};
