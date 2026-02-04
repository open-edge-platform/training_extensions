// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState, type Dispatch, type SetStateAction } from 'react';

import type { Label, Media, TaskType } from '../../constants/shared-types';
import { useLoadImageQuery } from '../../features/annotator/hooks/use-load-image-query.hook';
import type { ToolType } from '../../features/annotator/tools/interface';
import { isClassificationTask, isSegmentationTask } from '../../features/project/task-type-guards';
import { useProject } from '../../hooks/api/project.hook';
import type { RegionOfInterest } from '../types';
import { useProjectLabelsWithEmptyLabel } from './labels';

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
    mediaItem: Media;
    image: ImageData;
    roi: RegionOfInterest;
};

export const AnnotatorProviderContext = createContext<AnnotatorContext | null>(null);

const useSelectedLabel = () => {
    const { data: project } = useProject();
    const labels = useProjectLabelsWithEmptyLabel();
    const hasDefaultLabel = !isClassificationTask(project.task.task_type);
    const defaultLabel = hasDefaultLabel && labels.length > 0 ? labels[0].id : null;
    const [selectedLabelId, setSelectedLabelId] = useState<string | null>(defaultLabel);

    const selectedLabel: Label | null = labels.find(({ id }) => id === selectedLabelId) ?? null;

    return {
        selectedLabel,
        selectedLabelId,
        setSelectedLabelId,
        labels,
    };
};

const getDefaultTool = (taskType: TaskType | null): ToolType => {
    if (isClassificationTask(taskType)) {
        return 'selection';
    }

    if (isSegmentationTask(taskType)) {
        return 'polygon';
    }

    return 'bounding-box';
};

export const AnnotatorProvider = ({ mediaItem, children }: { mediaItem: Media; children: ReactNode }) => {
    const { data: selectedProject } = useProject();
    const [activeTool, setActiveTool] = useState<ToolType>(getDefaultTool(selectedProject?.task.task_type));

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
