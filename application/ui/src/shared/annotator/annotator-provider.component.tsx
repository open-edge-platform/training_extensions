// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, Dispatch, ReactNode, SetStateAction, useContext, useState } from 'react';

import { DatasetItem } from 'src/constants/shared-types';

import { useLoadImageQuery } from '../../features/annotator/hooks/use-load-image-query.hook';
import { ToolType } from '../../features/annotator/tools/interface';
import { RegionOfInterest } from '../../features/annotator/types';

type AnnotatorContext = {
    // Tools
    activeTool: ToolType | null;
    setActiveTool: Dispatch<SetStateAction<ToolType>>;

    // Media item
    mediaItem: DatasetItem;
    image: ImageData;
    roi: RegionOfInterest;
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

export const AnnotatorProvider = ({ mediaItem, children }: { mediaItem: DatasetItem; children: ReactNode }) => {
    const [activeTool, setActiveTool] = useState<ToolType>('selection');

    const imageQuery = useLoadImageQuery(mediaItem);

    return (
        <AnnotatorProviderContext.Provider
            value={{
                activeTool,
                setActiveTool,

                mediaItem,
                image: imageQuery.data,
                roi: { x: 0, y: 0, width: mediaItem.width, height: mediaItem.height },
            }}
        >
            {children}
        </AnnotatorProviderContext.Provider>
    );
};

export const useAnnotator = () => {
    const context = useContext(AnnotatorProviderContext);

    if (context === null) {
        throw new Error('useAnnotator was used outside of AnnotatorProvider');
    }

    return context;
};
