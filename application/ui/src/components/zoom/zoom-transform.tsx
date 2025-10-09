// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useRef, useState } from 'react';

import { clampBetween } from '@geti/smart-tools/utils';
import { createUseGesture, dragAction, pinchAction, wheelAction } from '@use-gesture/react';

import { useContainerSize } from './use-container-size';
import { usePanning } from './use-panning.hook';
import { Size, useSyncZoom } from './use-sync-zoom.hook';
import { getZoomState } from './util';
import { useSetZoom, useZoom } from './zoom.provider';

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
    const { setZoom } = useSetZoom();
    const isPanning = usePanning();
    const [isGrabbing, setIsGrabbing] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);
    const containerSize = useContainerSize(containerRef);
    useSyncZoom({ container: containerSize, zoomInMultiplier, zoomOutDivisor, target });

    const cursorIcon = isPanning && isGrabbing ? 'grabbing' : isPanning ? 'grab' : 'default';

    useGesture(
        {
            onPinch: ({ origin, offset: [deltaDistance] }) => {
                const rect = containerRef.current?.getBoundingClientRect();
                if (!rect) return;

                const factor = 1 + deltaDistance / 200;
                const newScale = clampBetween(
                    zoom.initialCoordinates.scale,
                    zoom.initialCoordinates.scale * factor,
                    zoom.maxZoomIn
                );
                const relativeCursor = { x: origin[0] - rect.left, y: origin[1] - rect.top };

                setZoom(
                    getZoomState({
                        newScale,
                        cursorX: relativeCursor.x,
                        cursorY: relativeCursor.y,
                        initialCoordinates: zoom.initialCoordinates,
                    })
                );
            },
            onWheel: ({ event, delta: [, verticalScrollDelta] }) => {
                const rect = containerRef.current?.getBoundingClientRect();
                if (!rect) return;

                const factor = 1 - verticalScrollDelta / 500;
                const newScale = clampBetween(zoom.initialCoordinates.scale, zoom.scale * factor, zoom.maxZoomIn);
                const relativeCursor = { x: event.clientX - rect.left, y: event.clientY - rect.top };

                setZoom(
                    getZoomState({
                        newScale,
                        cursorX: relativeCursor.x,
                        cursorY: relativeCursor.y,
                        initialCoordinates: zoom.initialCoordinates,
                    })
                );
            },
            onDrag: ({ delta: [dx, dy] }) => {
                setZoom((prev) => ({
                    ...prev,
                    translate: {
                        x: prev.translate.x + dx,
                        y: prev.translate.y + dy,
                    },
                }));
            },
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
            onPointerUp={() => setIsGrabbing(false)}
            onPointerDown={() => setIsGrabbing(true)}
        >
            <div
                data-testid='zoom-transform'
                className={classes.wrapperInternal}
                style={{
                    transformOrigin: '0 0',
                    transform: `translate(${zoom.translate.x}px, ${zoom.translate.y}px) scale(${zoom.scale})`,
                }}
            >
                {children}
            </div>
        </div>
    );
};
