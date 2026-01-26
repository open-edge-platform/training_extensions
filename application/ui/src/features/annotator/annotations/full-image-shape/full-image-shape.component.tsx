// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Annotation } from '../../../../shared/types';
import { useSelectedData } from '../../../dataset/selected-data-provider.component';

type FullImageShapeProps = {
    annotation: Annotation;
};

export const FullImageShape = ({ annotation }: FullImageShapeProps) => {
    const { labels } = annotation;
    const { selectedMediaItem } = useSelectedData();
    const color = labels.length ? labels[0].color : '--annotation-fill';

    return (
        <rect
            fill={'none'}
            stroke={color}
            aria-label='annotation rect'
            width={selectedMediaItem?.width}
            height={selectedMediaItem?.height}
        />
    );
};
