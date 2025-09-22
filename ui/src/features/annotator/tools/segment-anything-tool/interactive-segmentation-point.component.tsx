// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useZoom } from '../../../../components/zoom/zoom';
import { InteractiveAnnotationPoint } from './segment-anything.interface';

interface InteractiveSegmentationPointProps extends InteractiveAnnotationPoint {
    isLoading: boolean;
}

export const InteractiveSegmentationPoint = ({ x, y, positive, isLoading }: InteractiveSegmentationPointProps) => {
    const { scale } = useZoom();
    const fill = positive ? 'var(--brand-moss)' : 'var(--brand-coral-cobalt)';
    const animationScale = 1 / scale;
    const pointRadius = 5 / scale;

    return (
        <>
            <circle
                cx={x}
                cy={y}
                r={pointRadius}
                aria-label={`${positive ? 'Positive' : 'Negative'} interactive segmentation point`}
                style={{
                    fill,
                    opacity: 'var(--markers-opacity)',
                    strokeWidth: 'calc(1.5px / var(--zoom-level))',
                    stroke: 'var(--spectrum-global-color-static-gray-100)',
                }}
                data-testid={`point-${positive ? 'positive' : 'negative'}`}
            />
            {isLoading && (
                <g
                    transform={`translate(${x}, ${y}) scale(${animationScale}, ${animationScale})`}
                    aria-label='Processing input'
                >
                    <path d='M 0 -20 A 20 20 00 0 1 0 20' stroke='white' strokeWidth='3' fillOpacity='0'>
                        <animateTransform
                            attributeName='transform'
                            attributeType='XML'
                            type='rotate'
                            dur='1s'
                            from='0 0 0'
                            to='360 0 0'
                            repeatCount='indefinite'
                        />
                    </path>
                </g>
            )}
        </>
    );
};
