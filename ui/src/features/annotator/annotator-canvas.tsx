// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, MouseEvent } from 'react';

import { Grid, View } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { ZoomProvider } from '../../components/zoom/zoom';
import { ZoomTransform } from '../../components/zoom/zoom-transform';
import { useProjectIdentifier } from '../../hooks/use-project-identifier.hook';
import { getImageUrl } from '../dataset/gallery/utils';
import { Annotations } from './annotations/annotations.component';
import { useSelectedAnnotations } from './select-annotation-provider.component';
import { ToolManager } from './tools/tool-manager.component';
import { Annotation, DatasetItem } from './types';

const DEFAULT_ANNOTATION_STYLES = {
    fillOpacity: 0.4,
    fill: 'var(--annotation-fill)',
    stroke: 'var(--annotation-stroke)',
    strokeLinecap: 'round',
    strokeWidth: 'calc(1px / var(--zoom-scale))',
    strokeDashoffset: 0,
    strokeDasharray: 0,
    strokeOpacity: 'var(--annotation-border-opacity, 1)',
} satisfies CSSProperties;

type AnnotatorCanvasProps = {
    mediaItem: DatasetItem;
    isFocussed: boolean;
};
export const AnnotatorCanvas = ({ mediaItem, isFocussed }: AnnotatorCanvasProps) => {
    const { setSelectedAnnotations } = useSelectedAnnotations();
    const project_id = useProjectIdentifier();
    const size = { width: mediaItem.width, height: mediaItem.height };
    // todo: pass media annotations
    const annotations: Annotation[] = [];

    const handleClickOutside = (e: MouseEvent<SVGSVGElement>): void => {
        if (e.target === e.currentTarget) {
            setSelectedAnnotations(new Set());
        }
    };

    return (
        <ZoomProvider>
            <ZoomTransform target={size}>
                <Grid areas={['innercanvas']} width={'100%'} height='100%'>
                    <View gridArea={'innercanvas'}>
                        <img src={getImageUrl(project_id, String(mediaItem.id))} alt='Collected data' />
                    </View>

                    {!isEmpty(annotations) && (
                        <View gridArea={'innercanvas'}>
                            <svg
                                width={size.width}
                                height={size.height}
                                style={DEFAULT_ANNOTATION_STYLES}
                                onClick={handleClickOutside}
                            >
                                <Annotations width={size.width} height={size.height} isFocussed={isFocussed} />

                                <ToolManager />
                            </svg>
                        </View>
                    )}
                </Grid>
            </ZoomTransform>
        </ZoomProvider>
    );
};
