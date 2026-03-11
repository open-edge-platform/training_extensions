// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, useContext, useState } from 'react';

import { useOverlayTriggerState } from '@react-stately/overlays';
import { OverlayTriggerState } from 'react-stately';

import { ImportDatasetToProjectState } from '../import-export/import-dataset/util';

type ImportDatasetDialogStateContextProps = {
    datasetImportDialogState: OverlayTriggerState;
    currentStagedId: string | null;
    setCurrentStagedId: Dispatch<React.SetStateAction<string | null>>;
    currentStep: ImportDatasetToProjectState;
    setCurrentStep: Dispatch<React.SetStateAction<ImportDatasetToProjectState>>;
};

const ImportDatasetDialogStateContext = createContext<ImportDatasetDialogStateContextProps | undefined>(undefined);

export const ImportDatasetDialogStateProvider = ({ children }: { children: ReactNode }) => {
    const datasetImportDialogState = useOverlayTriggerState({});
    const [currentStep, setCurrentStep] = useState<ImportDatasetToProjectState>('uploading');
    const [currentStagedId, setCurrentStagedId] = useState<string | null>(null);

    return (
        <ImportDatasetDialogStateContext.Provider
            value={{ datasetImportDialogState, currentStagedId, setCurrentStagedId, currentStep, setCurrentStep }}
        >
            {children}
        </ImportDatasetDialogStateContext.Provider>
    );
};

export const useImportDatasetDialogState = () => {
    const context = useContext(ImportDatasetDialogStateContext);

    if (context === undefined) {
        throw new Error('useImportDatasetDialogState was used outside of ImportDatasetDialogStateProvider');
    }

    return context;
};
