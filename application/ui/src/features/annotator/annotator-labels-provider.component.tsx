// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useContext, useState } from 'react';

import { useProject } from 'hooks/api/project.hook';

import type { Label } from '../../constants/shared-types';
import { useProjectLabelsWithEmptyLabel } from '../../shared/annotator/labels';
import { isClassificationTask } from '../project/task-type-guards';

type AnnotatorLabelsContextProps = {
    labels: Label[];
    selectedLabel: Label | null;
    selectedLabelId: string | null;
    setSelectedLabelId: (id: string | null) => void;
};

const AnnotatorLabelsContext = createContext<AnnotatorLabelsContextProps | null>(null);

const useInitialSelectedLabelId = (labels: Label[]): string | null => {
    const { data: project } = useProject();
    const hasDefaultLabel = !isClassificationTask(project.task.task_type);

    return hasDefaultLabel && labels.length > 0 ? labels[0].id : null;
};

type AnnotatorLabelsProviderProps = {
    children: ReactNode;
};

export const AnnotatorLabelsProvider = ({ children }: AnnotatorLabelsProviderProps) => {
    const labels = useProjectLabelsWithEmptyLabel();
    const initialSelectedLabelId = useInitialSelectedLabelId(labels);
    const [selectedLabelId, setSelectedLabelId] = useState<string | null>(initialSelectedLabelId);

    const selectedLabel: Label | null = labels.find(({ id }) => id === selectedLabelId) ?? null;

    return (
        <AnnotatorLabelsContext value={{ labels, selectedLabel, selectedLabelId, setSelectedLabelId }}>
            {children}
        </AnnotatorLabelsContext>
    );
};

export const useAnnotatorLabels = (): AnnotatorLabelsContextProps => {
    const context = useContext(AnnotatorLabelsContext);

    if (context === null) {
        throw new Error('useAnnotatorLabels was used outside of AnnotatorLabelsProvider');
    }

    return context;
};
