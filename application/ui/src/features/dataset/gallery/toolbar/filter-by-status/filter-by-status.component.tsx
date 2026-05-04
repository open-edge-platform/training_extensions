// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';

import { FilterByStatusKey } from '../../../../../constants/shared-types';

const FILTER_BY_STATUS_OPTIONS: { name: string; key: FilterByStatusKey }[] = [
    { name: 'All media', key: 'all' },
    { name: 'Media with annotations', key: 'with_annotations' },
    { name: 'Media with missing annotations', key: 'missing_annotations' },
];

export const FilterByStatus = () => {
    const { annotationStatus, setAnnotationStatus } = useDatasetFiltersSearchParams();

    return (
        <Picker
            maxWidth='size-3000'
            aria-label={'media status'}
            items={FILTER_BY_STATUS_OPTIONS}
            selectedKey={annotationStatus ?? FILTER_BY_STATUS_OPTIONS[0].key}
            onSelectionChange={(status) => setAnnotationStatus(status as FilterByStatusKey)}
        >
            {(item) => <Item>{item.name}</Item>}
        </Picker>
    );
};
