// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useMemo, useState } from 'react';

import { useGetActiveModel } from '../models/hooks/api/use-get-active-model.hook';
import { useGetSuccessfulModels } from '../models/hooks/api/use-get-models.hook';
import { getAllModelsWithOpenVinoVariants, SelectableModel } from '../models/utils';

type PredictionsSetupContextProps = {
    selectableModels: SelectableModel[];
    selectedModelId: string | undefined;
    selectedModel: SelectableModel | undefined;
    changeSelectedModelId: (modelId: string | undefined) => void;
};

const PredictionSetupContext = createContext<PredictionsSetupContextProps | null>(null);

export const PredictionsSetupProvider = ({ children }: { children: ReactNode }) => {
    const { data: models } = useGetSuccessfulModels();
    const activeModel = useGetActiveModel();

    const selectableModels = useMemo(() => getAllModelsWithOpenVinoVariants(models), [models]);

    const defaultSelectedId =
        selectableModels.find((m) => m.modelId === activeModel?.model_variant_id)?.id ?? selectableModels.at(0)?.id;

    const [selectedModelId, setSelectedModelId] = useState<string | undefined>(defaultSelectedId);

    const selectedModel = selectableModels.find((model) => model.id === selectedModelId);

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
