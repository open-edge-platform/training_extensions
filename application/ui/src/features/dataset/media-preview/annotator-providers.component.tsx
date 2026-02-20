// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, Suspense } from 'react';

import { Flex, Loading } from '@geti/ui';

import { ZoomProvider } from '../../../components/zoom/zoom.provider';
import type { AnnotationDTO, Media } from '../../../constants/shared-types';
import { AnnotationActionsProvider } from '../../../shared/annotator/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../shared/annotator/annotation-visibility-provider.component';
import { SelectAnnotationProvider } from '../../../shared/annotator/select-annotation-provider.component';
import { LabelsProvider } from '../../annotator/labels-provider.component';
import {
    MediaItemImageLoader,
    SelectedMediaItemProvider,
} from '../../annotator/selected-media-item-provider.component';
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

const CanvasAreaLoading = () => (
    <Flex gridArea={'canvas'} alignContent={'center'} justifyContent={'center'}>
        <Loading size='L' mode='inline' />
    </Flex>
);

/**
 * Provider tree for the annotator.
 *
 * Stable providers (Zoom, SelectAnnotation, Visibility, CanvasSettings, Labels,
 * SelectedMediaItem, AnnotationActions) are placed OUTSIDE the Suspense boundary
 * so that toolbars and annotation state remain mounted while a new image loads.
 *
 * MediaItemImageLoader is the only suspending provider and lives INSIDE the
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
                        <LabelsProvider key={mode}>
                            <SelectedMediaItemProvider mediaItem={mediaItem}>
                                <AnnotationActionsProvider
                                    key={mediaItem.id}
                                    mediaItem={mediaItem}
                                    initialAnnotationsDTO={initialAnnotationsDTO}
                                    initialPredictionsDTO={initialPredictionsDTO}
                                    isUserReviewed={isUserReviewed}
                                    mode={mode}
                                >
                                    <Suspense fallback={<CanvasAreaLoading />}>
                                        <MediaItemImageLoader>{children}</MediaItemImageLoader>
                                    </Suspense>
                                </AnnotationActionsProvider>
                            </SelectedMediaItemProvider>
                        </LabelsProvider>
                    </CanvasSettingsProvider>
                </AnnotationVisibilityProvider>
            </SelectAnnotationProvider>
        </ZoomProvider>
    );
};
