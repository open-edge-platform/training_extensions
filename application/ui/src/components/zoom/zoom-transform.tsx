// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useRef } from 'react';

import { createUseGesture, pinchAction, wheelAction } from '@use-gesture/react';

import { useContainerSize } from './use-container-size';
import { Size, useSyncZoom } from './use-sync-zoom.hook';
import { useSetZoom, useZoom, ZoomState } from './zoom.provider';

import classes from './zoom.module.scss';

const useGesture = createUseGesture([wheelAction, pinchAction]);

export const ZoomTransform = ({ children, target }: { children: ReactNode; target: Size }) => {
    const zoom = useZoom();
    const setZoom = useSetZoom();
    const containerRef = useRef<HTMLDivElement>(null);
    const containerSize = useContainerSize(containerRef);
    const initialCoordinates = useSyncZoom({ container: containerSize, target });
    const maxZoomIn = initialCoordinates.scale * 100;
    const maxZoomOut = initialCoordinates.scale / 2;

    useGesture(
        {
            onPinch: ({ origin, offset: [deltaDistance] }) => {
                const rect = containerRef.current?.getBoundingClientRect();
                if (!rect) return;

                const factor = 1 + deltaDistance / 200;
                const newScale = getClampScale(initialCoordinates.scale * factor);
                const relativeCursor = { x: origin[0] - rect.left, y: origin[1] - rect.top };

                setZoom(getZoomState(newScale, relativeCursor.x, relativeCursor.y));
            },
            onWheel: ({ event, delta: [, verticalScrollDelta] }) => {
                const rect = containerRef.current?.getBoundingClientRect();
                if (!rect) return;

                const factor = 1 - verticalScrollDelta / 500;
                const newScale = getClampScale(zoom.scale * factor);
                const relativeCursor = { x: event.clientX - rect.left, y: event.clientY - rect.top };

                setZoom(getZoomState(newScale, relativeCursor.x, relativeCursor.y));
            },
        },
        {
            target: containerRef,
            eventOptions: { passive: false },
            wheel: { preventDefault: true },
            pinch: { preventDefault: true },
        }
    );

    const getClampScale = (value: number) => {
        return Math.max(maxZoomOut, Math.min(maxZoomIn, value));
    };

    const getZoomState = (newScale: number, cursorX: number, cursorY: number) => (prev: ZoomState) => {
        if (newScale <= initialCoordinates.scale) {
            return { ...prev, ...initialCoordinates };
        }

        const scaleRatio = newScale / prev.scale;
        const newTranslateX = cursorX - scaleRatio * (cursorX - prev.translate.x);
        const newTranslateY = cursorY - scaleRatio * (cursorY - prev.translate.y);

        return {
            ...prev,
            scale: newScale,
            translate: { x: newTranslateX, y: newTranslateY },
        };
    };

    return (
        <div
            ref={containerRef}
            className={classes.wrapper}
            style={{ transform: 'translate3d(0, 0, 0)', '--zoom-scale': zoom.scale }}
        >
            <div
                data-testid='zoom-transform'
                className={classes.wrapperInternal}
                style={{
                    transformOrigin: '0 0',
                    transform: `translate(${zoom.translate.x}px, ${zoom.translate.y}px) scale(var(--zoom-scale))`,
                }}
            >
                {children}
            </div>
        </div>
    );
};
