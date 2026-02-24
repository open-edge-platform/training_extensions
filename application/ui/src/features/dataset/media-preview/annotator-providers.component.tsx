// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { ZoomProvider } from '../../../components/zoom/zoom.provider';
import type { AnnotationDTO, Media } from '../../../constants/shared-types';
import { AnnotationActionsProvider } from '../../../shared/annotator/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../shared/annotator/annotation-visibility-provider.component';
import { SelectAnnotationProvider } from '../../../shared/annotator/select-annotation-provider.component';
import { AnnotatorLabelsProvider } from '../../annotator/annotator-labels-provider.component';
import { SelectedMediaItemProvider } from '../../annotator/selected-media-item-provider.component';
import { CanvasSettingsProvider } from './primary-toolbar/settings/canvas-settings-provider.component';
import type { AnnotatorMode } from './secondary-toolbar/annotator-modes/mode';

type AnnotatorProvidersProps = {
    mediaItem: Media;
    initialAnnotationsDTO: AnnotationDTO[];
    initialPredictionsDTO: AnnotationDTO[];
    isUserReviewed: boolean;
    mode: AnnotatorMode;
    children: ReactNode;
};

/**
 * Provider tree for the annotator.
 *
 * Stable providers (Zoom, SelectAnnotation, Visibility, CanvasSettings, Labels,
 * SelectedMediaItem, AnnotationActions) are placed OUTSIDE the Suspense boundary
 * so that toolbars and annotation state remain mounted while a new image loads.
 *
 * MediaItemImageLoader (media-preview.component.tsx) is the only suspending provider and lives INSIDE the
 * Suspense boundary — meaning only the canvas area shows a loading state during
 * image fetching.
 */
export const AnnotatorProviders = ({
    mediaItem,
    initialAnnotationsDTO,
    initialPredictionsDTO,
    isUserReviewed,
    mode,
    children,
}: AnnotatorProvidersProps) => {
    return (
        <ZoomProvider>
            <SelectAnnotationProvider>
                <AnnotationVisibilityProvider>
                    <CanvasSettingsProvider>
                        <AnnotatorLabelsProvider>
                            <SelectedMediaItemProvider mediaItem={mediaItem}>
                                <AnnotationActionsProvider
                                    key={mediaItem.id}
                                    mediaItem={mediaItem}
                                    initialAnnotationsDTO={initialAnnotationsDTO}
                                    initialPredictionsDTO={initialPredictionsDTO}
                                    isUserReviewed={isUserReviewed}
                                    mode={mode}
                                >
                                    {children}
                                </AnnotationActionsProvider>
                            </SelectedMediaItemProvider>
                        </AnnotatorLabelsProvider>
                    </CanvasSettingsProvider>
                </AnnotationVisibilityProvider>
            </SelectAnnotationProvider>
        </ZoomProvider>
    );
};
