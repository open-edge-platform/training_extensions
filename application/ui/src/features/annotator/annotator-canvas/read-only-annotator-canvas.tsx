// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { MouseEvent, useRef } from 'react';

import { Loading } from '@geti/ui';
import { useIsFetching } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { useSpinDelay } from 'spin-delay';

import { ZoomTransform } from '../../../components/zoom/zoom-transform';
import type { Media } from '../../../constants/shared-types';
import { ReadOnlyAnnotations } from '../annotations/read-only-annotations.component';
import { loadImageQueryOptions } from '../hooks/use-load-image-query.hook';
import { MediaImage } from './media-image.component';

type ReadOnlyAnnotatorCanvasProps = {
    mediaItem: Media;
    image: ImageData;
};

export const ReadOnlyAnnotatorCanvas = ({ mediaItem, image }: ReadOnlyAnnotatorCanvasProps) => {
    const projectId = useProjectIdentifier();
    const containerRef = useRef<HTMLDivElement>(null);

    const isFetchingMedia = useIsFetching({ queryKey: loadImageQueryOptions(projectId, mediaItem).queryKey }) > 0;
    const isLoadingMedia = useSpinDelay(isFetchingMedia, { delay: 400, minDuration: 200 });

    const isPlaceholderImage = image.width === 1 && image.height === 1;
    const size = { width: mediaItem.width, height: mediaItem.height };

    if (isLoadingMedia && isPlaceholderImage) {
        return <Loading size='M' />;
    }

    return (
        <ZoomTransform target={size}>
            <div
                ref={containerRef}
                style={{ position: 'relative', height: '100%', width: '100%' }}
                onContextMenu={(event: MouseEvent): void => event.preventDefault()}
            >
                {isLoadingMedia && <Loading mode={'overlay'} />}
                <MediaImage image={image} mediaItem={mediaItem} />
                <ReadOnlyAnnotations width={size.width} height={size.height} />
            </div>
        </ZoomTransform>
    );
};
