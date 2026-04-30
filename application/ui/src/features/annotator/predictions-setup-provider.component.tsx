// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useMemo, useState } from 'react';

import { useGetActiveModel } from '../models/hooks/api/use-get-active-model.hook';
import { useGetSuccessfulModels } from '../models/hooks/api/use-get-models.hook';
import { getAllModelsWithOpenVinoVariants, SelectableModel } from '../models/utils';

type PredictionsSetupContextProps = {
    models: SelectableModel[];
    selectedModelId: string | undefined;
    selectedModel: SelectableModel | undefined;
    changeSelectedModelId: (modelId: string | undefined) => void;
};

const PredictionSetupContext = createContext<PredictionsSetupContextProps | null>(null);

export const PredictionsSetupProvider = ({ children }: { children: ReactNode }) => {
    const { data: models } = useGetSuccessfulModels();

    const activeModel = useGetActiveModel();
    const [selectedModelId, setSelectedModelId] = useState<string | undefined>(activeModel?.id ?? models.at(0)?.id);

    const allSelectableModels = useMemo(() => getAllModelsWithOpenVinoVariants(models), [models]);
    const selectedModel = allSelectableModels.find((model) => model.id === selectedModelId);

    return (
        <PredictionSetupContext
            value={{
                selectedModelId,
                selectedModel,
                changeSelectedModelId: setSelectedModelId,
                models: allSelectableModels,
            }}
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
