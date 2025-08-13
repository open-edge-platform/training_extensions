import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { Selection } from '@geti/ui';

type SelectedDataState = null | {
    selectedKeys: Selection;
    setSelectedKeys: Dispatch<SetStateAction<Selection>>;
};

export const SelectedDataContext = createContext<SelectedDataState>(null);

export const SelectedDataProvider = ({ children }: { children: ReactNode }) => {
    const [selectedKeys, setSelectedKeys] = useState<Selection>(new Set());

    return (
        <SelectedDataContext.Provider value={{ selectedKeys, setSelectedKeys }}>
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
