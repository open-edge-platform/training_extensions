// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key, Selection } from '@geti/ui';

import { Media } from '../../../../constants/shared-types';
import { isVideo } from '../../../../shared/media-item-utils';

export const toggleMultipleSelection =
    (items: Key[]) =>
    (selectedItems: Selection): Selection => {
        if (selectedItems === 'all') {
            return new Set();
        }

        const allItemsSelected = selectedItems.size === items.length;
        const someItemsSelected = selectedItems.size > 0 && !allItemsSelected;

        if (selectedItems.size === 0 || someItemsSelected) {
            return new Set(items);
        }

        return new Set();
    };

export const getNumberOfImagesAndVideosMessage = (mediaItems: Media[], numberOfItems: number) => {
    const numberOfVideos = mediaItems.filter(isVideo).length;
    const numberOfImages = numberOfItems - numberOfVideos;

    const imagesMessage = `${numberOfImages} Item${numberOfImages === 1 ? '' : 's'}`;
    const videosMessage = `${numberOfVideos} video${numberOfVideos == 1 ? '' : 's'}`;

    if (numberOfImages > 0 && numberOfVideos > 0) {
        return `${imagesMessage}, ${videosMessage}`;
    }

    if (numberOfImages > 0) {
        return imagesMessage;
    }

    if (numberOfVideos > 0) {
        return videosMessage;
    }

    return '';
};
