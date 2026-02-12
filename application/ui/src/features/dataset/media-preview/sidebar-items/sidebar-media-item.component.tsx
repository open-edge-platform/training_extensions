// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { MediaItem } from '../../../../components/media-item/media-item.component';
import { MediaThumbnail } from '../../../../components/media-thumbnail/media-thumbnail.component';
import type { Media } from '../../../../constants/shared-types';
import { getThumbnailUrl } from '../../../../shared/media-url.utils';

type SidebarMediaItemProps = {
    item: Media;
    onSelectedMediaItem: (item: Media) => void;
};

export const SidebarMediaItem = ({ item, onSelectedMediaItem }: SidebarMediaItemProps) => {
    const projectId = useProjectIdentifier();

    return (
        <MediaItem
            contentElement={() => (
                <MediaThumbnail
                    item={item}
                    alt={item.name}
                    url={getThumbnailUrl(projectId, String(item.id))}
                    onClick={() => onSelectedMediaItem(item)}
                />
            )}
        />
    );
};
