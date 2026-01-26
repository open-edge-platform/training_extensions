// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { View } from '@geti/ui';
import { Accept, Search } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { MediaItem } from '../../../../components/media-item/media-item.component';
import { MediaThumbnail } from '../../../../components/media-thumbnail/media-thumbnail.component';
import type { Media } from '../../../../constants/shared-types';
import { useAnnotationActions } from '../../../../shared/annotator/annotation-actions-provider.component';
import { getThumbnailUrl } from '../../../../shared/media-url.utils';

import classes from './sidebar-media-item.module.scss';

type SidebarMediaItemProps = {
    item: Media;
    isSelected: boolean;
    onSelectedMediaItem: (item: Media) => void;
};

export const SidebarMediaItem = ({ item, isSelected, onSelectedMediaItem }: SidebarMediaItemProps) => {
    const project_id = useProjectIdentifier();
    const { isUserReviewed } = useAnnotationActions();

    return (
        <MediaItem
            contentElement={() => (
                <MediaThumbnail
                    alt={item.name}
                    url={getThumbnailUrl(project_id, String(item.id))}
                    onClick={() => onSelectedMediaItem(item)}
                />
            )}
            bottomRightElement={() => {
                if (!isSelected) {
                    return null;
                }

                return (
                    <View UNSAFE_className={isUserReviewed ? classes.iconAccept : classes.iconSearch}>
                        {isUserReviewed ? <Accept /> : <Search />}
                    </View>
                );
            }}
        />
    );
};
