// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext } from 'react';

import { UndoRedoActions } from './undo-redo-actions.interface';

const UndoRedoContext = createContext<UndoRedoActions | undefined>(undefined);

interface UndoRedoProviderProps {
    children: ReactNode;
    state: UndoRedoActions;
}

export const UndoRedoProvider = ({ state, children }: UndoRedoProviderProps) => {
    return <UndoRedoContext.Provider value={state}>{children}</UndoRedoContext.Provider>;
};

export const useParentUndoRedo = (): UndoRedoActions | undefined => {
    return useContext(UndoRedoContext);
};

export const useUndoRedo = (): UndoRedoActions => {
    const context = useParentUndoRedo();

    if (context === undefined) {
        throw new Error('useUndoRedo must be used within an UndoRedoProvider');
    }

    return context;
};
