// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

type SelectedAnnotationContextProps = {
    selectedAnnotations: Set<string>;
    setSelectedAnnotations: Dispatch<SetStateAction<Set<string>>>;
};

const SelectedAnnotationContext = createContext<SelectedAnnotationContextProps | null>(null);

export const useSelectedAnnotations = () => {
    const ctx = useContext(SelectedAnnotationContext);

    if (ctx === null) {
        throw new Error('No context');
    }

    return ctx;
};

export const SelectAnnotationProvider = ({ children }: { children: ReactNode }) => {
    const [selectedAnnotations, setSelectedAnnotations] = useState<Set<string>>(new Set());

    return (
        <SelectedAnnotationContext.Provider value={{ selectedAnnotations, setSelectedAnnotations }}>
            {children}
        </SelectedAnnotationContext.Provider>
    );
};
