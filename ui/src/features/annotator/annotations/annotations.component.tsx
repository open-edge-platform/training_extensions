// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';

import { useAnnotator } from '../annotator-provider.component';
import { useSelectedAnnotations } from '../select-annotation-provider.component';
import { ToolManager } from '../tools/tool-manager.component';
import { DEFAULT_ANNOTATION_STYLES } from '../utils';
import { Annotation } from './annotation.component';
import { MaskAnnotations } from './mask-annotations.component';

type AnnotationsProps = {
    width: number;
    height: number;
    isFocussed: boolean;
};

export const Annotations = ({ width, height, isFocussed }: AnnotationsProps) => {
    const { annotations } = useAnnotator();
    const { selectedAnnotations } = useSelectedAnnotations();

    // Order annotations by selection. Selected annotation should always be on top.
    const orderedAnnotations = [
        ...annotations.filter((a) => !selectedAnnotations.has(a.id)),
        ...annotations.filter((a) => selectedAnnotations.has(a.id)),
    ];

    return (
        <svg width={width} height={height} style={DEFAULT_ANNOTATION_STYLES}>
            {!isEmpty(annotations) && (
                <MaskAnnotations annotations={orderedAnnotations} width={width} height={height} isEnabled={isFocussed}>
                    {orderedAnnotations.map((annotation) => (
                        <Annotation annotation={annotation} key={annotation.id} />
                    ))}
                </MaskAnnotations>
            )}
            <ToolManager />
        </svg>
    );
};
