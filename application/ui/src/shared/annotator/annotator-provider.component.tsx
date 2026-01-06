// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState, type Dispatch, type SetStateAction } from 'react';

import { useProjectLabels } from 'hooks/use-project-labels.hook';

import type { DatasetItem, Label } from '../../constants/shared-types';
import { useLoadImageQuery } from '../../features/annotator/hooks/use-load-image-query.hook';
import type { ToolType } from '../../features/annotator/tools/interface';
import type { RegionOfInterest } from '../../features/annotator/types';

type AnnotatorContext = {
    // Tools
    activeTool: ToolType | null;
    setActiveTool: Dispatch<SetStateAction<ToolType>>;

    // Labels
    selectedLabelId: string | null;
    setSelectedLabelId: (id: string | null) => void;
    selectedLabel: Label | null;
    labels: Label[];

    // Media item
    mediaItem: DatasetItem;
    image: ImageData;
    roi: RegionOfInterest;
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

const useSelectedLabel = () => {
    const labels = useProjectLabels();

    const [selectedLabelId, setSelectedLabelId] = useState<string | null>(labels.length === 0 ? null : labels[0].id);

    const selectedLabel: Label | null = labels.find(({ id }) => id === selectedLabelId) ?? null;

    return {
        selectedLabel,
        selectedLabelId,
        setSelectedLabelId,
        labels,
    };
};

export const AnnotatorProvider = ({ mediaItem, children }: { mediaItem: DatasetItem; children: ReactNode }) => {
    const [activeTool, setActiveTool] = useState<ToolType>('selection');

    const { selectedLabel, selectedLabelId, setSelectedLabelId, labels } = useSelectedLabel();

    const imageQuery = useLoadImageQuery(mediaItem);

    return (
        <AnnotatorProviderContext.Provider
            value={{
                activeTool,
                setActiveTool,

                setSelectedLabelId,
                selectedLabelId,
                selectedLabel,
                labels,

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
