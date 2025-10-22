// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { API_BASE_URL } from 'src/api/client';
import { DatasetItem } from 'src/constants/shared-types';

import { ZoomTransform } from '../../components/zoom/zoom-transform';
import { useAnnotationActions } from '../../shared/annotator/annotation-actions-provider.component';
import { useAnnotationVisibility } from '../../shared/annotator/annotation-visibility-provider.component';
import { useSelectedAnnotations } from '../../shared/annotator/select-annotation-provider.component';
import { Annotations } from './annotations/annotations.component';
import { ToolManager } from './tools/tool-manager.component';

const getImageUrl = (projectId: string, itemId: string) => {
    return `${API_BASE_URL}/api/projects/${projectId}/dataset/items/${itemId}/binary`;
};

type AnnotatorCanvasProps = {
    mediaItem: DatasetItem;
};

export const AnnotatorCanvas = ({ mediaItem }: AnnotatorCanvasProps) => {
    const project_id = useProjectIdentifier();
    const { annotations } = useAnnotationActions();
    const { selectedAnnotations } = useSelectedAnnotations();
    const { isFocussed } = useAnnotationVisibility();

    // Order annotations by selection. Selected annotation should always be on top.
    const orderedAnnotations = [
        ...annotations.filter((a) => !selectedAnnotations.has(a.id)),
        ...annotations.filter((a) => selectedAnnotations.has(a.id)),
    ];

    const size = { width: mediaItem.width, height: mediaItem.height };

    return (
        <ZoomTransform target={size}>
            <View position={'relative'} width={'100%'} height={'100%'}>
                <img src={getImageUrl(project_id, String(mediaItem.id))} alt='Collected data' />

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
