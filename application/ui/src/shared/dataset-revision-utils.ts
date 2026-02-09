// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { DatasetRevisionItem, Media } from '../constants/shared-types';

/**
 * Converts a DatasetRevisionItem to Media format.
 *
 * Dataset revision items are snapshots of media items used for training.
 * They share the same ID as the original media item, allowing us to fetch
 * annotations from the main dataset endpoint.
 *
 * This conversion adds the required Media fields (name, type, size, source_id)
 * that are needed by the annotator components.
 *
 * NOTE: In the future the whole media/datasetItem/datasetRevisionItem will hopefully be
 * merged and we wont need multiple endpoints to get the full dataset of the item.
 */
export const datasetRevisionItemToMedia = (item: DatasetRevisionItem): Media => {
    return {
        id: item.id,
        name: item.id, // Use ID as name since revision items don't have names
        type: 'image',
        format: item.format ?? ('jpg' as 'jpg' | 'png'),
        width: item.width ?? 1,
        height: item.height ?? 1,
        size: 1, // Revision items don't track file size
        source_id: null,
    };
};
