// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext } from 'react';

import { useOverlayTriggerState } from '@react-stately/overlays';
import { OverlayTriggerState } from 'react-stately';

interface ImportDatasetDialogStateContextProps {
    datasetImportDialogState: OverlayTriggerState;
}

const ImportDatasetDialogStateContext = createContext<ImportDatasetDialogStateContextProps | undefined>(undefined);

export const ImportDatasetDialogStateProvider = ({ children }: { children: ReactNode }) => {
    const datasetImportDialogState = useOverlayTriggerState({});

    return (
        <ImportDatasetDialogStateContext.Provider value={{ datasetImportDialogState }}>
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
