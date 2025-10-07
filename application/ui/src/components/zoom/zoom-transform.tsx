// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useRef } from 'react';

import { createUseGesture, pinchAction, wheelAction } from '@use-gesture/react';

import { useContainerSize } from './use-container-size';
import { Size, useSyncZoom } from './use-sync-zoom.hook';
import { useSetZoom, useZoom } from './zoom.provider';

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
    console.log('initialCoordinates', initialCoordinates.translate);

    useGesture(
        {
            onWheel: ({ event, delta: [, dy] }) => {
                event.preventDefault();

                const rect = containerRef.current?.getBoundingClientRect();
                if (!rect) return;

                const isZoomingOut = dy > 0;
                const factor = 1 - dy / 500;
                const newScale = Math.max(maxZoomOut, Math.min(maxZoomIn, zoom.scale * factor));

                const cursorX = event.clientX - rect.left;
                const cursorY = event.clientY - rect.top;

                setZoom((prev) => {
                    if (isZoomingOut && newScale <= initialCoordinates.scale) {
                        return { ...prev, ...initialCoordinates };
                    }

                    const scaleRatio = newScale / prev.scale;
                    const newTranslateX = cursorX - scaleRatio * (cursorX - prev.translate.x);
                    const newTranslateY = cursorY - scaleRatio * (cursorY - prev.translate.y);

                    return {
                        ...prev,
                        scale: newScale,
                        translate: {
                            x: newTranslateX,
                            y: newTranslateY,
                        },
                    };
                });
            },
        },
        {
            target: containerRef,
            eventOptions: { passive: false },
            wheel: { preventDefault: true },
        }
    );

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
                    transition: 'transform 100ms ease',
                }}
            >
                {children}
            </div>
        </div>
    );
};
