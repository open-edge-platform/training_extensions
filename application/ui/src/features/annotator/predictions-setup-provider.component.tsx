// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState } from 'react';

import { Model } from '../../constants/shared-types';
import { useGetActiveModel } from '../models/hooks/api/use-get-active-model.hook';
import { useGetSuccessfulModels } from '../models/hooks/api/use-get-models.hook';

type PredictionsSetupContextProps = {
    models: Model[];
    selectedModelId: string | undefined;
    changeSelectedModelId: (modelId: string | undefined) => void;
};

const PredictionSetupContext = createContext<PredictionsSetupContextProps | null>(null);

export const PredictionsSetupProvider = ({ children }: { children: ReactNode }) => {
    const { data: models } = useGetSuccessfulModels();

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
