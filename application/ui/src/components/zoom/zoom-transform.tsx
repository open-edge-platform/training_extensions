// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useCallback, useRef } from 'react';

import { clampBetween } from '@geti/smart-tools/utils';
import { createUseGesture, dragAction, pinchAction, wheelAction } from '@use-gesture/react';

import { Point } from './types';
import { useContainerSize } from './use-container-size';
import { usePanning } from './use-panning.hook';
import { Size, useSyncZoom } from './use-sync-zoom.hook';
import { useWheelPanning } from './use-wheel-panning.hook';
import { useSetZoom, useZoom, zoomStore } from './zoom.store';

import classes from './zoom.module.scss';

type ZoomTransformProps = {
    target: Size;
    children: ReactNode;
    zoomOutDivisor?: number;
    zoomInMultiplier?: number;
};

const useGesture = createUseGesture([wheelAction, pinchAction, dragAction]);

export const ZoomTransform = ({ children, target, zoomInMultiplier = 10, zoomOutDivisor = 2 }: ZoomTransformProps) => {
    const zoom = useZoom();
    const { setZoom, zoomToCursor } = useSetZoom();
    const { isPanning, setIsPanning } = usePanning();
    const containerRef = useRef<HTMLDivElement>(null);
    const containerSize = useContainerSize(containerRef);
    const { onPointerDown, onPointerUp, onPointerMove, onMouseLeave, isGrabbing } = useWheelPanning(setIsPanning);

    useSyncZoom({ container: containerSize, zoomInMultiplier, zoomOutDivisor, target });

    const cursorIcon = isPanning && isGrabbing ? 'grabbing' : isPanning ? 'grab' : 'default';

    const handleTranslateUpdate = useCallback(
        ({ x, y }: Point) => {
            const currentZoom = zoomStore.getState();
            setZoom({
                hasAnimation: false,
                translate: { x: currentZoom.translate.x + x, y: currentZoom.translate.y + y },
            });
        },
        [setZoom]
    );

    useGesture(
        {
            onPinch: ({ origin, offset: [deltaDistance] }) => {
                const rect = containerRef.current?.getBoundingClientRect();
                if (!rect) return;

                const currentZoom = zoomStore.getState();
                const factor = 1 + deltaDistance / 200;
                const newScale = clampBetween(
                    currentZoom.initialCoordinates.scale,
                    currentZoom.initialCoordinates.scale * factor,
                    currentZoom.maxZoomIn
                );
                const relativeCursor = { x: origin[0] - rect.left, y: origin[1] - rect.top };

                zoomToCursor(newScale, relativeCursor.x, relativeCursor.y);
            },
            onWheel: ({ event, delta: [, verticalScrollDelta] }) => {
                const rect = containerRef.current?.getBoundingClientRect();
                if (!rect) return;

                const currentZoom = zoomStore.getState();
                const factor = 1 - verticalScrollDelta / 500;
                const newScale = clampBetween(
                    currentZoom.initialCoordinates.scale,
                    currentZoom.scale * factor,
                    currentZoom.maxZoomIn
                );
                const relativeCursor = { x: event.clientX - rect.left, y: event.clientY - rect.top };

                zoomToCursor(newScale, relativeCursor.x, relativeCursor.y);
            },
            onDrag: ({ delta: [x, y] }) => handleTranslateUpdate({ x, y }),
        },
        {
            target: containerRef,
            eventOptions: { passive: false },
            wheel: { preventDefault: true },
            pinch: { preventDefault: true },
            drag: { enabled: isPanning },
        }
    );

    return (
        <div
            ref={containerRef}
            className={classes.wrapper}
            style={{
                cursor: cursorIcon,
                touchAction: 'none',
                transform: 'translate3d(0, 0, 0)',
                '--zoom-scale': zoom.scale,
            }}
            onPointerMove={onPointerMove(handleTranslateUpdate)}
            onPointerDown={onPointerDown}
            onPointerUp={onPointerUp}
            onMouseLeave={onMouseLeave}
        >
            <div
                data-testid='zoom-transform'
                className={classes.wrapperInternal}
                style={{
                    transformOrigin: '0 0',
                    transition: zoom.hasAnimation ? 'transform 0.2s ease' : 'none',
                    transform: `translate(${zoom.translate.x}px, ${zoom.translate.y}px) scale(${zoom.scale})`,
                }}
            >
                {children}
            </div>
        </div>
    );
};
