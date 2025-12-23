// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import { View } from '@geti/ui';
import type { DatasetItem } from 'src/constants/shared-types';
import { useAnnotator } from 'src/shared/annotator/annotator-provider.component';

import { ZoomTransform } from '../../components/zoom/zoom-transform';
import { useAnnotationActions } from '../../shared/annotator/annotation-actions-provider.component';
import { useAnnotationVisibility } from '../../shared/annotator/annotation-visibility-provider.component';
import { useSelectedAnnotations } from '../../shared/annotator/select-annotation-provider.component';
import { Annotations } from './annotations/annotations.component';
import { ToolManager } from './tools/tool-manager.component';

import styles from './annotator-canvas.module.scss';

type AnnotatorCanvasProps = {
    mediaItem: DatasetItem;
};

export const AnnotatorCanvas = ({ mediaItem }: AnnotatorCanvasProps) => {
    const { annotations } = useAnnotationActions();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { isFocussed } = useAnnotationVisibility();
    const { image } = useAnnotator();

    const drawImageOnCanvas = useCallback(
        (canvasRef: HTMLCanvasElement | null) => {
            if (!canvasRef) return;

            canvasRef.width = image.width;
            canvasRef.height = image.height;

            const ctx = canvasRef.getContext('2d');
            if (ctx) {
                ctx.putImageData(image, 0, 0);
            }
        },
        [image]
    );

    // Order annotations by selection. Selected annotation should always be on top.
    const orderedAnnotations = [
        ...annotations.filter((a) => !selectedAnnotations.has(a.id)),
        ...annotations.filter((a) => selectedAnnotations.has(a.id)),
    ];

    const size = { width: mediaItem.width, height: mediaItem.height };

    return (
        <ZoomTransform target={size}>
            <View position={'relative'} width={'100%'} height={'100%'}>
                <canvas aria-label='Captured frame' ref={drawImageOnCanvas} className={styles.image} />

                <Annotations
                    annotations={orderedAnnotations}
                    width={size.width}
                    height={size.height}
                    isFocussed={isFocussed}
                />
                <ToolManager />
            </View>
        </ZoomTransform>
    );
};
