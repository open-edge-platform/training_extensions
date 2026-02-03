// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, ReactNode } from 'react';

import { Anchor as InternalAnchor } from './anchor.component';

export const ANCHOR_SIZE = 8;

interface ResizeAnchorProps {
    zoom: number;
    x: number;
    y: number;
    label: string;
    fill?: string;
    stroke?: string;
    Anchor?: ReactNode;
    cursor?: CSSProperties['cursor'];
    strokeWidth?: number;
    contextMenu?: (onClose: () => void) => ReactNode;
    onStart?: () => void;
    onComplete: () => void;
    moveAnchorTo: (x: number, y: number) => void;
}

export const ResizeAnchor = ({
    x,
    y,
    zoom,
    onStart,
    onComplete,
    moveAnchorTo,
    label,
    contextMenu,
    fill = 'white',
    cursor = 'all-scroll',
    stroke = 'var(--energy-blue)',
    strokeWidth = 1,
}: ResizeAnchorProps) => {
    const size = ANCHOR_SIZE / zoom;

    // We render both a visual anchor and an invisible anchor that has a larger
    // clicking area than the visible one
    const visualAnchorProps = {
        fill,
        stroke,
        strokeWidth: strokeWidth / zoom,
    };

    return (
        <InternalAnchor
            size={size}
            label={label}
            x={x}
            y={y}
            zoom={zoom}
            fill={fill}
            cursor={cursor ? cursor : 'default'}
            onStart={onStart}
            onComplete={onComplete}
            moveAnchorTo={moveAnchorTo}
            contextMenu={contextMenu}
        >
            <rect
                fillOpacity={1.0}
                style={{ transformOrigin: `${x}px ${y}px` }}
                x={x - size / 2}
                y={y - size / 2}
                width={size}
                height={size}
                {...visualAnchorProps}
            />
        </InternalAnchor>
    );
};
