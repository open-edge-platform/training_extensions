// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import { useLocalStorage } from 'usehooks-ts';

const PINNED_LABELS_KEY = 'pinned-labels';

const getPinnedLabelsKey = (projectId: string) => {
    return `${PINNED_LABELS_KEY}-${projectId}`;
};

type UsePinnedLabelsReturn = {
    pinnedLabelIds: string[];
    isPinned: (labelId: string) => boolean;
    togglePin: (labelId: string) => void;
    hasPinnedLabels: boolean;
};

export const usePinnedLabels = (projectId: string): UsePinnedLabelsReturn => {
    const [pinnedLabelIds, setPinnedLabelIds] = useLocalStorage<string[]>(getPinnedLabelsKey(projectId), []);

    const isPinned = useCallback(
        (labelId: string) => {
            return pinnedLabelIds.includes(labelId);
        },
        [pinnedLabelIds]
    );

    const togglePin = (labelId: string) => {
        setPinnedLabelIds((prev) => {
            if (prev.includes(labelId)) {
                return prev.filter((id) => id !== labelId);
            }
            return [...prev, labelId];
        });
    };

    const hasPinnedLabels = pinnedLabelIds.length > 0;

    return { pinnedLabelIds, isPinned, togglePin, hasPinnedLabels };
};
