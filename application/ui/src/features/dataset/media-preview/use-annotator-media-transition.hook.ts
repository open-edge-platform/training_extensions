// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import type { Media } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import { useSelectedMediaItem } from '../../annotator/selected-media-item-provider.component';

type UseAnnotatorMediaTransitionProps = {
    onSelectedMediaItem: (item: Media) => void;
};
export const useAnnotatorMediaTransition = ({ onSelectedMediaItem }: UseAnnotatorMediaTransitionProps) => {
    const { setMediaItem } = useSelectedMediaItem();
    const { setSelectedAnnotations } = useSelectedAnnotations();
    const { resetAnnotations } = useAnnotationActions();

    return useCallback(
        (item: Media) => {
            setSelectedAnnotations(new Set());
            resetAnnotations();
            setMediaItem(item);
            onSelectedMediaItem(item);
        },
        [onSelectedMediaItem, resetAnnotations, setMediaItem, setSelectedAnnotations]
    );
};
