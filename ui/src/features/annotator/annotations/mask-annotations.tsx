// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useId } from 'react';

type MaskAnnotationsProps = {
    children: ReactNode;
    width: number;
    height: number;
    isEnabled: boolean;
};

export const MaskAnnotations = ({ children, width, height, isEnabled }: MaskAnnotationsProps) => {
    const id = useId();

    const maskOpacity = isEnabled ? 0.8 : 0.0;

    return (
        <>
            <mask id={`mask-${id}`}>
                <rect x='0' y='0' width={width} height={height} style={{ fill: 'white', fillOpacity: 1.0 }} />
            </mask>
            <rect
                x={0}
                y={0}
                width={width}
                height={height}
                mask={`url(#mask-${id})`}
                style={{
                    fillOpacity: maskOpacity,
                    fill: 'black',
                    strokeWidth: 0,
                    transition: 'fill-opacity 0.1s ease-in-out',
                    transitionDelay: isEnabled ? '0s' : '.25s',
                    transitionDuration: isEnabled ? '0.2s' : '0.1s',
                }}
            />
            <g>{children}</g>
        </>
    );
};
