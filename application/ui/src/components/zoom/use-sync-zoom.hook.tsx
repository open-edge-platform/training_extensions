// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo } from 'react';

import { useSetZoom } from './zoom.provider';

export type Size = { width: number; height: number };

const DEFAULT_SCREEN_ZOOM = 1;
const getCenterCoordinates = (container: Size, target: Size) => {
    // Scale image so that it fits perfectly in the container
    const scale = DEFAULT_SCREEN_ZOOM * Math.min(container.width / target.width, container.height / target.height);

    return {
        scale,
        // Center image, considering scale, transform origin is the top-left corner
        translate: {
            x: (container.width - target.width * scale) / 2,
            y: (container.height - target.height * scale) / 2,
        },
    };
};

const INITIAL_ZOOM = { scale: 1.0, translate: { x: 0, y: 0 } };
export const useSyncZoom = ({ container, target }: { container: Size; target: Size }) => {
    const setZoom = useSetZoom();

    const targetZoom = useMemo(() => {
        if (container.width === undefined || container.height === undefined) {
            return INITIAL_ZOOM;
        }

        return getCenterCoordinates({ width: container.width, height: container.height }, target);
    }, [container, target]);

    useEffect(() => {
        const scale = Number(targetZoom.scale.toFixed(3));

        setZoom({
            scale,
            min: scale / 2,
            max: scale * 2,
            translate: {
                x: Number(targetZoom.translate.x.toFixed(3)),
                y: Number(targetZoom.translate.y.toFixed(3)),
            },
        });
    }, [targetZoom.scale, targetZoom.translate.x, targetZoom.translate.y, setZoom]);

    return targetZoom;
};
