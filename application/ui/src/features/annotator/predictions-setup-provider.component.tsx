// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useMemo } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { orderBy } from 'lodash-es';
import { useLocalStorage } from 'usehooks-ts';

import { Model } from '../../constants/shared-types';
import { useGetActiveModel } from '../models/hooks/api/use-get-active-model.hook';
import { useGetSuccessfulModels } from '../models/hooks/api/use-get-models.hook';
import { getAllModelsWithOpenVINOVariants, SelectableModel } from '../models/utils';

type PredictionsSetupContextProps = {
    selectableModels: SelectableModel[];
    selectedModelId: string | null;
    selectedModel: SelectableModel | undefined;
    changeSelectedModelId: (modelId: string | null) => void;
};

const PredictionSetupContext = createContext<PredictionsSetupContextProps | null>(null);

const getLatestModel = (models: Model[]): string | null => {
    const sortedModels = orderBy(models, (model) => model.training_info.end_time, 'desc');

    return getAllModelsWithOpenVINOVariants(sortedModels).at(0)?.modelVariantId ?? null;
};

const useSelectedModelId = (models: Model[]) => {
    const projectId = useProjectIdentifier();
    const activeModel = useGetActiveModel();

    const selectableModels = useMemo(() => getAllModelsWithOpenVINOVariants(models), [models]);

    const defaultSelectedId =
        selectableModels.find((model) => model.modelVariantId === activeModel?.model_variant_id)?.modelVariantId ??
        getLatestModel(models);

    return useLocalStorage<string | null>(`${projectId}-model-variant-id`, defaultSelectedId);
};

export const PredictionsSetupProvider = ({ children }: { children: ReactNode }) => {
    const { data: models } = useGetSuccessfulModels();

    const selectableModels = useMemo(() => getAllModelsWithOpenVINOVariants(models), [models]);

    const [selectedModelId, setSelectedModelId] = useSelectedModelId(models);

    const selectedModel = selectableModels.find((model) => model.modelVariantId === selectedModelId);

    return (
        <PredictionSetupContext
            value={{ selectedModelId, selectedModel, changeSelectedModelId: setSelectedModelId, selectableModels }}
        >
            {children}
        </PredictionSetupContext>
    );
};

export const usePredictionSetup = () => {
    const context = useContext(PredictionSetupContext);

    if (context === null) {
        throw new Error('usePredictionSetup was used outside of PredictionsSetupProvider');
    }

    return context;
};
