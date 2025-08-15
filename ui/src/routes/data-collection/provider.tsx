import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { Selection } from '@geti/ui';

export type AnnotationState = 'rejected' | 'accepted';
export type MediaState = Map<string, AnnotationState>;

type SelectedDataState = null | {
    selectedKeys: Selection;
    setSelectedKeys: Dispatch<SetStateAction<Selection>>;

    mediaState: MediaState;
    setMediaState: Dispatch<SetStateAction<MediaState>>;
};

export const SelectedDataContext = createContext<SelectedDataState>(null);

export const SelectedDataProvider = ({ children }: { children: ReactNode }) => {
    const [mediaState, setMediaState] = useState<MediaState>(new Map());
    const [selectedKeys, setSelectedKeys] = useState<Selection>(new Set());

    return (
        <SelectedDataContext.Provider value={{ selectedKeys, setSelectedKeys, mediaState, setMediaState }}>
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
