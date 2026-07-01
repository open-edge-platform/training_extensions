// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Divider, Flex } from '@geti-ui/ui';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { useProjectLabels } from 'hooks/use-project-labels.hook';
import { isEmpty } from 'lodash-es';

import type { DatasetItemAnnotationStatus, Label } from '../../../../constants/shared-types';
import { formatDateRangeEnd, formatDateRangeStart } from '../../../../shared/date-utils';
import { FilterChips } from '../toolbar/filter-chips/filter-chips.component';

const ANNOTATION_STATUS_LABELS: Record<DatasetItemAnnotationStatus, string> = {
    with_annotations: 'Media with annotations',
    missing_annotations: 'Media with missing annotations',
};

export const ActiveFilters = () => {
    const labels = useProjectLabels();
    const {
        selectedLabelIds,
        setSelectedLabelIds,
        annotationStatus,
        setAnnotationStatus,
        startDate,
        setStartDate,
        endDate,
        setEndDate,
    } = useDatasetFiltersSearchParams();

    const selectedLabels = selectedLabelIds
        .map((id) => labels.find((label) => label.id === id))
        .filter(Boolean) as Label[];

    const hasActiveFilters =
        !isEmpty(selectedLabelIds) || annotationStatus !== null || startDate !== null || endDate !== null;

    if (!hasActiveFilters) {
        return null;
    }

    const handleRemoveLabel = (id: string) => {
        setSelectedLabelIds(selectedLabelIds.filter((selectedId) => selectedId !== id));
    };

    const handleClearAll = () => {
        setSelectedLabelIds([]);
        setAnnotationStatus(null);
        setStartDate(null);
        setEndDate(null);
    };

    return (
        <Flex gap={'size-150'} wrap={'wrap'} alignItems={'center'} aria-label='Active filters'>
            <ActionButton isQuiet onPress={handleClearAll}>
                Clear all
            </ActionButton>

            <Divider orientation={'vertical'} size={'S'} />

            {selectedLabels.map((label) => (
                <FilterChips key={label.id} name={label.name} onClose={() => handleRemoveLabel(label.id)} />
            ))}

            {annotationStatus !== null && (
                <FilterChips
                    name={ANNOTATION_STATUS_LABELS[annotationStatus]}
                    onClose={() => setAnnotationStatus(null)}
                />
            )}

            {startDate !== null && (
                <FilterChips name={formatDateRangeStart(startDate)} onClose={() => setStartDate(null)} />
            )}

            {endDate !== null && <FilterChips name={formatDateRangeEnd(endDate)} onClose={() => setEndDate(null)} />}
        </Flex>
    );
};
