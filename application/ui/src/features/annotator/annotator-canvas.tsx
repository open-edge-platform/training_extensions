// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { ZoomTransform } from '../../components/zoom/zoom-transform';
import { getImageUrl } from '../dataset/gallery/utils';
import { useAnnotationActions } from './annotation-actions-provider.component';
import { Annotations } from './annotations/annotations.component';
import { useSelectedAnnotations } from './select-annotation-provider.component';
import { ToolManager } from './tools/tool-manager.component';
import { DatasetItem } from './types';

type AnnotatorCanvasProps = {
    mediaItem: DatasetItem;
    isFocussed: boolean;
};
export const AnnotatorCanvas = ({ mediaItem, isFocussed }: AnnotatorCanvasProps) => {
    const project_id = useProjectIdentifier();
    const { annotations } = useAnnotationActions();
    const { selectedAnnotations } = useSelectedAnnotations();

    // Order annotations by selection. Selected annotation should always be on top.
    const orderedAnnotations = [
        ...annotations.filter((a) => !selectedAnnotations.has(a.id)),
        ...annotations.filter((a) => selectedAnnotations.has(a.id)),
    ];

    const size = { width: mediaItem.width, height: mediaItem.height };

    return (
        <ZoomTransform target={size}>
            <div style={{ width: '100%', height: '100%', position: 'relative' }}>
                <img src={getImageUrl(project_id, String(mediaItem.id))} alt='Collected data' />
                <Annotations
                    annotations={orderedAnnotations}
                    width={size.width}
                    height={size.height}
                    isFocussed={isFocussed}
                />
                <ToolManager />
            </div>
        </ZoomTransform>
    );
};
