// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { PointerEvent } from 'react';

import { Grid, View } from '@geti/ui';

import { ZoomProvider } from '../../components/zoom/zoom';
import { ZoomTransform } from '../../components/zoom/zoom-transform';
import { useProjectIdentifier } from '../../hooks/use-project-identifier.hook';
import { getImageUrl } from '../dataset/gallery/utils';
import { Annotations } from './annotations/annotations.component';
import { useSelectedAnnotations } from './select-annotation-provider.component';
import { ToolManager } from './tools/tool-manager.component';
import { Annotation, DatasetItem } from './types';

type AnnotatorCanvasProps = {
    mediaItem: DatasetItem;
    isFocussed: boolean;
};
export const AnnotatorCanvas = ({ mediaItem, isFocussed }: AnnotatorCanvasProps) => {
    const project_id = useProjectIdentifier();
    const { setSelectedAnnotations } = useSelectedAnnotations();

    const size = { width: mediaItem.width, height: mediaItem.height };
    // TODO: pass media annotations
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const annotations: Annotation[] = [];

    const handleClickOutside = (e: PointerEvent<HTMLDivElement>): void => {
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

                    <View gridArea={'innercanvas'}>
                        <>
                            <div onPointerDown={handleClickOutside}>
                                <Annotations width={size.width} height={size.height} isFocussed={isFocussed} />
                            </div>
                            <ToolManager />
                        </>
                    </View>
                </Grid>
            </ZoomTransform>
        </ZoomProvider>
    );
};
