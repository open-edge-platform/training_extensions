// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MouseEvent, useCallback } from 'react';

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { API_BASE_URL } from '../../../api/client';
import { ZoomTransform } from '../../../components/zoom/zoom-transform';
import type { Media } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useAnnotationVisibility } from '../../../shared/annotator/annotation-visibility-provider.component';
import { useAnnotator } from '../../../shared/annotator/annotator-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import { Annotations } from '../annotations/annotations.component';
import { ToolManager } from '../tools/tool-manager.component';

import classes from './annotator-canvas.module.scss';

type AnnotatorCanvasProps = {
    mediaItem: Media;
};

export const AnnotatorCanvas = ({ mediaItem }: AnnotatorCanvasProps) => {
    const { annotations } = useAnnotationActions();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { isFocussed } = useAnnotationVisibility();
    const { image } = useAnnotator();

    const drawImageOnCanvas = useCallback(
        (ref: HTMLCanvasElement | null) => {
            const ctx = ref?.getContext('2d');

            if (ctx == null) {
                return;
            }

            ctx.putImageData(image, 0, 0);
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
            <div
                style={{ position: 'relative', height: '100%', width: '100%' }}
                onContextMenu={(event: MouseEvent): void => event.preventDefault()}
            >
                <canvas ref={drawImageOnCanvas} width={image.width} height={image.height} className={classes.image} />

                <Annotations
                    width={size.width}
                    height={size.height}
                    isFocussed={isFocussed}
                    annotations={orderedAnnotations}
                />
                <ToolManager />
            </div>
        </ZoomTransform>
    );
};
