import { Key, Selection } from '@geti/ui';

import { AnnotationState, MediaState } from '../../../routes/data-collection/provider';

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

export const updateSelectedKeysTo =
    (selectedKeys: Selection, annotationState: AnnotationState) => (map: MediaState) => {
        const newMap = new Map(map.entries());

        if (selectedKeys === 'all') {
            return newMap;
        }

        selectedKeys.forEach((mediaId) => newMap.set(String(mediaId), annotationState));

        return newMap;
    };
