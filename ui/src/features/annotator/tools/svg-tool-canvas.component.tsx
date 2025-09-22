// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FC, MouseEvent, PropsWithChildren, RefObject, SVGProps } from 'react';

import { roiFromImage } from '@geti/smart-tools/utils';

import { useSelectedAnnotations } from '../select-annotation-provider.component';
import { allowPanning } from '../utils';

type CanvasProps = SVGProps<SVGSVGElement> & { image: ImageData } & { canvasRef?: RefObject<SVGRectElement | null> };
// This svg component is used to by tools that need to add local listeners that work in
// a given region of interest.
// An invisible rect is rendered to guarantee that the svg gets a width and height.
export const SvgToolCanvas: FC<PropsWithChildren<CanvasProps>> = ({
    image,
    children,
    canvasRef,
    onPointerDown,
    ...props
}) => {
    const { setSelectedAnnotations } = useSelectedAnnotations();

    const roi = roiFromImage(image);

    const handleClickOutside = (e: MouseEvent<SVGSVGElement>): void => {
        if (e.target === e.currentTarget) {
            setSelectedAnnotations(new Set());
        }
    };

    return (
        <svg
            {...props}
            onClick={handleClickOutside}
            onPointerDown={allowPanning(onPointerDown)}
            // eslint-disable-next-line jsx-a11y/aria-role
            role='editor'
            viewBox={`0 0 ${roi.width} ${roi.height}`}
        >
            <rect {...roi} fillOpacity={0} ref={canvasRef} />
            {children}
        </svg>
    );
};
