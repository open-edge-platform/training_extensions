// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useHotkeys } from 'react-hotkeys-hook';

import { useZoom } from '../../../components/zoom/zoom.provider';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';
import { useSelectedAnnotations } from '../../../shared/annotator/select-annotation-provider.component';
import { HOTKEYS } from '../../../shared/hotkeys-definition';
import { EditBoundingBox } from '../tools/edit-bounding-box/edit-bounding-box.component';
import { EditPolygon } from '../tools/edit-polygon/edit-polygon.component';
import { SvgToolCanvas } from '../tools/svg-tool-canvas.component';
import { isPolygon, isRectangle } from './utils';

interface EditableSelectedAnnotationsProps {
    image: ImageData;
}

export const EditableSelectedAnnotations = ({ image }: EditableSelectedAnnotationsProps) => {
    const { scale } = useZoom();
    const { annotations } = useAnnotationActions();
    const { deleteAnnotations } = useAnnotationActions();
    const { selectedAnnotations, setSelectedAnnotations } = useSelectedAnnotations();

    const selected =
        selectedAnnotations.size === 0
            ? []
            : annotations.filter((annotation) => selectedAnnotations.has(annotation.id));

    const handleDeleteAnnotations = () => {
        if (selectedAnnotations.size === 0) {
            return;
        }

        const annotationsToDelete = Array.from(selectedAnnotations);

        setSelectedAnnotations(new Set());
        deleteAnnotations(annotationsToDelete);
    };

    useHotkeys([HOTKEYS.deleteAnnotation, HOTKEYS.deleteAnnotationAlternative], handleDeleteAnnotations);

    useHotkeys(
        HOTKEYS.selectAllAnnotations,
        (event) => {
            event.preventDefault();

            setSelectedAnnotations((prev) => {
                return new Set([...prev, ...annotations.map((annotation) => annotation.id)]);
            });
        },
        [setSelectedAnnotations, annotations]
    );

    useHotkeys(
        HOTKEYS.deselectAllAnnotations,
        (event) => {
            event.preventDefault();

            setSelectedAnnotations(() => {
                return new Set();
            });
        },
        [setSelectedAnnotations]
    );

    if (selected.length === 0) {
        return null;
    }

    return (
        <SvgToolCanvas
            image={image}
            width={image.width}
            height={image.height}
            style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}
            aria-label={'selected annotations editor'}
        >
            {selected.map((annotation) => {
                if (isPolygon(annotation)) {
                    return <EditPolygon key={annotation.id} annotation={annotation} zoom={scale} />;
                }

                if (isRectangle(annotation)) {
                    const { shape } = annotation;

                    return (
                        <EditBoundingBox
                            key={`box-${annotation.id}-${shape.x}-${shape.y}-${shape.width}-${shape.height}`}
                            annotation={annotation}
                            zoom={scale}
                        />
                    );
                }

                return null;
            })}
        </SvgToolCanvas>
    );
};
