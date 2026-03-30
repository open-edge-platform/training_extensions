// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState } from 'react';

import { useGetActiveModel } from '../models/hooks/api/use-get-active-model.hook';
import { useGetModels } from '../models/hooks/api/use-get-models.hook';
import { isSuccessfulModel } from '../models/model-listing/utils/utils';

type PredictionsSetupContextProps = {
    models: { name: string; id: string }[];
    selectedModelId: string | undefined;
    changeSelectedModelId: (modelId: string | undefined) => void;
};

const PredictionSetupContext = createContext<PredictionsSetupContextProps | null>(null);

export const PredictionsSetupProvider = ({ children }: { children: ReactNode }) => {
    const { data } = useGetModels();
    const models = data.filter((model) => isSuccessfulModel(model) && !model.files_deleted);

    const activeModel = useGetActiveModel();
    const [selectedModelId, setSelectedModelId] = useState<string | undefined>(activeModel?.id ?? models.at(0)?.id);

    return (
        <PredictionSetupContext value={{ selectedModelId, changeSelectedModelId: setSelectedModelId, models }}>
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
