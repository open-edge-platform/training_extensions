// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState } from 'react';

import { useProject } from 'hooks/api/project.hook';

import type { Label } from '../../constants/shared-types';
import { useProjectLabelsWithEmptyLabel } from '../../shared/annotator/labels';
import { isClassificationTask } from '../project/task-type-guards';

type LabelsContextType = {
    labels: Label[];
    selectedLabel: Label | null;
    selectedLabelId: string | null;
    setSelectedLabelId: (id: string | null) => void;
};

const LabelsContext = createContext<LabelsContextType | null>(null);

const useInitialSelectedLabelId = (labels: Label[]): string | null => {
    const { data: project } = useProject();
    const hasDefaultLabel = !isClassificationTask(project.task.task_type);

    return hasDefaultLabel && labels.length > 0 ? labels[0].id : null;
};

type LabelsProviderProps = {
    children: ReactNode;
};

export const LabelsProvider = ({ children }: LabelsProviderProps) => {
    const labels = useProjectLabelsWithEmptyLabel();
    const initialSelectedLabelId = useInitialSelectedLabelId(labels);
    const [selectedLabelId, setSelectedLabelId] = useState<string | null>(initialSelectedLabelId);

    const selectedLabel: Label | null = labels.find(({ id }) => id === selectedLabelId) ?? null;

    return (
        <LabelsContext.Provider value={{ labels, selectedLabel, selectedLabelId, setSelectedLabelId }}>
            {children}
        </LabelsContext.Provider>
    );
};

export const useLabelsProvider = (): LabelsContextType => {
    const context = useContext(LabelsContext);

    if (context === null) {
        throw new Error('useLabelsProvider was used outside of LabelsProvider');
    }

    return context;
};
