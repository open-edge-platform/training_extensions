// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, ReactNode } from 'react';

import { Anchor as InternalAnchor } from './anchor.component';

export const ANCHOR_SIZE = 8;

enum ResizeAnchorType {
    SQUARE,
    CIRCLE,
    CUSTOM,
}

interface ResizeAnchorProps {
    zoom: number;
    x: number;
    y: number;
    moveAnchorTo: (x: number, y: number) => void;
    cursor?: CSSProperties['cursor'];
    label: string;
    onStart?: () => void;
    onComplete: () => void;
    type?: ResizeAnchorType;
    fill?: string;
    stroke?: string;
    strokeWidth?: number;
    Anchor?: ReactNode;
}

interface DefaultCircleProps {
    zoom: number;
    x: number;
    y: number;
    fill?: string;
    stroke?: string;
    strokeWidth?: number;
}

const DefaultCircle = ({ x, y, zoom, fill, stroke, strokeWidth = 1 }: DefaultCircleProps) => {
    return <circle cx={x} cy={y} r={ANCHOR_SIZE / zoom / 2} {...{ fill, stroke, strokeWidth: strokeWidth / zoom }} />;
};

export const ResizeAnchor = ({
    x,
    y,
    zoom,
    onStart,
    onComplete,
    moveAnchorTo,
    label,
    fill = 'white',
    type = ResizeAnchorType.SQUARE,
    cursor = 'all-scroll',
    stroke = 'var(--energy-blue)',
    strokeWidth = 1,
    Anchor = <DefaultCircle x={x} y={y} zoom={zoom} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />,
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
        >
            {type === ResizeAnchorType.SQUARE ? (
                <g fillOpacity={1.0} transform-origin={`${x}px ${y}px`}>
                    <rect x={x - size / 2} y={y - size / 2} width={size} height={size} {...visualAnchorProps} />
                </g>
            ) : (
                Anchor
            )}
        </InternalAnchor>
    );
};
