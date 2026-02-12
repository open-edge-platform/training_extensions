// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, useContext, useState, type Dispatch, type ReactNode, type SetStateAction } from 'react';

import type { TaskType } from '../../constants/shared-types';
import type { ToolType } from '../../features/annotator/tools/interface';
import { AnnotatorMode } from '../../features/dataset/media-preview/secondary-toolbar/annotator-modes/mode';
import { isClassificationTask, isSegmentationTask } from '../../features/project/task-type-guards';
import { useProject } from '../../hooks/api/project.hook';

type ToolContextValue = {
    activeTool: ToolType | null;
    setActiveTool: Dispatch<SetStateAction<ToolType | null>>;
};

const ToolContext = createContext<ToolContextValue | null>(null);

export const useTool = (): ToolContextValue => {
    const context = useContext(ToolContext);

    if (context === null) {
        throw new Error('useTool must be used within a ToolProvider');
    }

    return context;
};

const getDefaultTool = (taskType: TaskType | null, mode: AnnotatorMode): ToolType | null => {
    if (mode === 'prediction') {
        return null;
    }

    if (isClassificationTask(taskType)) {
        return null;
    }

    if (isSegmentationTask(taskType)) {
        return 'polygon';
    }

    return 'bounding-box';
};

type ToolProviderProps = {
    mode: AnnotatorMode;
    children: ReactNode;
};

export const ToolProvider = ({ mode, children }: ToolProviderProps) => {
    const { data: selectedProject } = useProject();
    const [activeTool, setActiveTool] = useState<ToolType | null>(() =>
        getDefaultTool(selectedProject.task.task_type, mode)
    );

    if (mode === 'prediction' && activeTool !== null) {
        setActiveTool(null);
    } else if (mode === 'annotation' && activeTool === null) {
        setActiveTool(getDefaultTool(selectedProject.task.task_type, mode));
    }

    return <ToolContext.Provider value={{ activeTool, setActiveTool }}>{children}</ToolContext.Provider>;
};
