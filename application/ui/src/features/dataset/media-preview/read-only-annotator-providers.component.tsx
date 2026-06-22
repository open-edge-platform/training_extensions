// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { ZoomProvider } from '../../../components/zoom/zoom.provider';
import type { AnnotationDTO, Media } from '../../../constants/shared-types';
import { AnnotationActionsProvider } from '../../../shared/annotator/annotation-actions-provider.component';
import { AnnotationVisibilityProvider } from '../../../shared/annotator/annotation-visibility-provider.component';
import { AnnotatorLabelsProvider } from '../../annotator/annotator-labels-provider.component';
import { CanvasSettingsProvider } from './primary-toolbar/settings/canvas-settings-provider.component';

type ReadOnlyAnnotatorProvidersProps = {
    mediaItem: Media;
    initialAnnotationsDTO: AnnotationDTO[];
    isUserReviewed: boolean;
    children: ReactNode;
};

export const ReadOnlyAnnotatorProviders = ({
    mediaItem,
    initialAnnotationsDTO,
    isUserReviewed,
    children,
}: ReadOnlyAnnotatorProvidersProps) => {
    return (
        <ZoomProvider>
            <AnnotationVisibilityProvider>
                <CanvasSettingsProvider>
                    <AnnotatorLabelsProvider>
                        <AnnotationActionsProvider
                            mediaItem={mediaItem}
                            initialAnnotationsDTO={initialAnnotationsDTO}
                            initialPredictionsDTO={[]}
                            isUserReviewed={isUserReviewed}
                            mode='annotation'
                            isReadOnly
                        >
                            {children}
                        </AnnotationActionsProvider>
                    </AnnotatorLabelsProvider>
                </CanvasSettingsProvider>
            </AnnotationVisibilityProvider>
        </ZoomProvider>
    );
};
