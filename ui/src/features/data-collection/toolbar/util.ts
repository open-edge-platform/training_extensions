import { Key, Selection } from '@geti/ui';

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
