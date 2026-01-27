// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { MediaItem } from '../../../../components/media-item/media-item.component';
import { MediaThumbnail } from '../../../../components/media-thumbnail/media-thumbnail.component';
import type { Media, MediaStateMap } from '../../../../constants/shared-types';
import { getThumbnailUrl } from '../../../../shared/media-url.utils';
import { AnnotationStatusIcon } from '../../gallery/annotation-state-icon.component';

type SidebarMediaItemProps = {
    item: Media;
    mediaState: MediaStateMap;
    onSelectedMediaItem: (item: Media) => void;
};

export const SidebarMediaItem = ({ item, mediaState, onSelectedMediaItem }: SidebarMediaItemProps) => {
    const projectId = useProjectIdentifier();
    const itemState = mediaState.get(String(item.id));

    return (
        <MediaItem
            contentElement={() => (
                <MediaThumbnail
                    alt={item.name}
                    url={getThumbnailUrl(projectId, String(item.id))}
                    onClick={() => onSelectedMediaItem(item)}
                />
            )}
            bottomRightElement={() => <AnnotationStatusIcon state={itemState} />}
        />
    );
};
