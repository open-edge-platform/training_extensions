// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useMemo, useState, type Dispatch, type SetStateAction } from 'react';

import { Selection } from '@geti-ui/ui';

type SelectedDataState = null | {
    selectedKeys: Selection;
    setSelectedKeys: Dispatch<SetStateAction<Selection>>;
    toggleSelectedKeys: (key: string[]) => void;
    isSelected: (key: string) => boolean;
};

const SelectedDataContext = createContext<SelectedDataState>(null);

export const SelectedDataProvider = ({ children }: { children: ReactNode }) => {
    const [selectedKeys, setSelectedKeys] = useState<Selection>(new Set());
    const selectedKeysSet = useMemo(() => new Set(selectedKeys), [selectedKeys]);

    const toggleSelectedKeys = (keys: string[]) => {
        setSelectedKeys((prevSelectedKeys) => {
            const updatedSelectedKeys = new Set(prevSelectedKeys);

            keys.forEach((key) => {
                updatedSelectedKeys.has(key) ? updatedSelectedKeys.delete(key) : updatedSelectedKeys.add(key);
            });

            return updatedSelectedKeys;
        });
    };

    const isSelected = (key: string) => {
        return selectedKeysSet.has(key);
    };

    return (
        <SelectedDataContext.Provider
            value={{
                selectedKeys,
                setSelectedKeys,
                toggleSelectedKeys,
                isSelected,
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
