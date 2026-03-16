// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useEffect, useState, type Dispatch, type SetStateAction } from 'react';

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

export const SelectAnnotationProvider = ({ children, resetKey }: { children: ReactNode; resetKey?: string }) => {
    const [selectedAnnotations, setSelectedAnnotations] = useState<Set<string>>(new Set());

    useEffect(() => {
        setSelectedAnnotations(new Set());
    }, [resetKey]);

    return (
        <SelectedAnnotationContext.Provider value={{ selectedAnnotations, setSelectedAnnotations }}>
            {children}
        </SelectedAnnotationContext.Provider>
    );
};
