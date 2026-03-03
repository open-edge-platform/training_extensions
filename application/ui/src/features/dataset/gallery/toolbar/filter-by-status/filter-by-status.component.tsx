// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Item, Picker } from '@geti/ui';

import { DatasetItemAnnotationStatus } from '../../../../../constants/shared-types';

export type FilterByStatusKey = 'all' | DatasetItemAnnotationStatus;

type FilterByStatusProps = {
    onChange: (status: FilterByStatusKey) => void;
};

const FILTER_BY_STATUS_OPTIONS: { name: string; key: FilterByStatusKey }[] = [
    { name: 'Status: All', key: 'all' },
    { name: 'Status: Unannotated', key: 'unannotated' },
    { name: 'Status: Reviewed', key: 'reviewed' },
    { name: 'Status: To Review', key: 'to_review' },
    { name: 'Status: Reviewed or Unannotated', key: 'reviewed_or_unannotated' },
];

export const FilterByStatus = ({ onChange }: FilterByStatusProps) => {
    return (
        <Picker
            maxWidth='size-3000'
            aria-label={'media status'}
            items={FILTER_BY_STATUS_OPTIONS}
            onSelectionChange={(status) => onChange(status as FilterByStatusKey)}
        >
            {(item) => <Item>{item.name}</Item>}
        </Picker>
    );
};
