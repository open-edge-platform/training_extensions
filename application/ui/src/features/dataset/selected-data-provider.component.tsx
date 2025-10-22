// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState, type Dispatch, type SetStateAction } from 'react';

import { Selection } from '@geti/ui';

export type AnnotationStatus = 'rejected' | 'accepted';
export type MediaState = Map<string, AnnotationStatus>;

type SelectedDataState = null | {
    selectedKeys: Selection;
    setSelectedKeys: Dispatch<SetStateAction<Selection>>;

    mediaState: MediaState;
    setMediaState: Dispatch<SetStateAction<MediaState>>;
    toggleSelectedKeys: (key: string[]) => void;
};

const SelectedDataContext = createContext<SelectedDataState>(null);

export const SelectedDataProvider = ({ children }: { children: ReactNode }) => {
    const [mediaState, setMediaState] = useState<MediaState>(new Map());
    const [selectedKeys, setSelectedKeys] = useState<Selection>(new Set());

    const toggleSelectedKeys = (keys: string[]) => {
        setSelectedKeys((prevSelectedKeys) => {
            const updatedSelectedKeys = new Set(prevSelectedKeys);

            keys.forEach((key) => {
                updatedSelectedKeys.has(key) ? updatedSelectedKeys.delete(key) : updatedSelectedKeys.add(key);
            });

            return updatedSelectedKeys;
        });
    };

    return (
        <SelectedDataContext.Provider
            value={{ selectedKeys, setSelectedKeys, mediaState, setMediaState, toggleSelectedKeys }}
        >
            {children}
        </SelectedDataContext.Provider>
    );
};

export const useSelectedData = () => {
    const context = useContext(SelectedDataContext);

    if (context === null) {
        throw new Error('useSelectedData was used outside of SelectedDataProvider');
    }

    return context;
};
