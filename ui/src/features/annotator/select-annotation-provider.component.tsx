// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { noop } from 'lodash-es';

import { Annotation } from './types';

type SelectedAnnotations = Set<Annotation>;

const SelectedAnnotation = createContext<SelectedAnnotations | null>(null);
const SetSelectedAnnotation = createContext<Dispatch<SetStateAction<SelectedAnnotations | null>>>(noop);

export const useSetSelectedAnnotations = () => {
    return useContext(SetSelectedAnnotation);
};

export const useSelectedAnnotation = () => {
    const ctx = useContext(SelectedAnnotation);

    if (ctx === null) {
        throw new Error('No context');
    }

    return ctx;
};

export const SelectAnnotationProvider = ({ children }: { children: ReactNode }) => {
    const [selectedAnnotations, setSelectedAnnotations] = useState<SelectedAnnotations | null>(new Set());

    return (
        <SelectedAnnotation.Provider value={selectedAnnotations}>
            <SetSelectedAnnotation.Provider value={setSelectedAnnotations}>{children}</SetSelectedAnnotation.Provider>
        </SelectedAnnotation.Provider>
    );
};
