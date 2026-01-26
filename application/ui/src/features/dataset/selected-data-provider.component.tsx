// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState, type Dispatch, type SetStateAction } from 'react';

import { Selection } from '@geti/ui';

import { MediaStateMap, type Media } from '../../constants/shared-types';

type SelectedDataState = null | {
    selectedKeys: Selection;
    setSelectedKeys: Dispatch<SetStateAction<Selection>>;

    mediaState: MediaStateMap;
    setMediaState: Dispatch<SetStateAction<MediaStateMap>>;
    toggleSelectedKeys: (key: string[]) => void;

    selectedMediaItem: Media | null;
    onSelectedMediaItemChange: (item: Media | null) => void;
};

const SelectedDataContext = createContext<SelectedDataState>(null);

export const SelectedDataProvider = ({ children }: { children: ReactNode }) => {
    const [mediaState, setMediaState] = useState<MediaStateMap>(new Map());
    const [selectedKeys, setSelectedKeys] = useState<Selection>(new Set());
    const [selectedMediaItem, setSelectedMediaItem] = useState<null | Media>(null);

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
            value={{
                selectedKeys,
                setSelectedKeys,
                mediaState,
                setMediaState,
                toggleSelectedKeys,
                selectedMediaItem,
                onSelectedMediaItemChange: setSelectedMediaItem,
            }}
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
