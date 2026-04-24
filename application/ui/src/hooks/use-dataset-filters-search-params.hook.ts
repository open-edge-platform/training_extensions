// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty, isNil } from 'lodash-es';
import { useSearchParams } from 'react-router-dom';
import { parse, stringify } from 'zipson/lib';

import type { DatasetItemAnnotationStatus, FilterByStatusKey } from '../constants/shared-types';
import { isNonEmptyString } from '../shared/util';

export const LABELS_PARAM = 'labelsFilter';
export const ANNOTATION_STATUS_PARAM = 'annotationStatusFilter';
export const START_DATE_PARAM = 'startDateFilter';
export const END_DATE_PARAM = 'endDateFilter';

const VALID_ANNOTATION_STATUSES = new Set<DatasetItemAnnotationStatus>(['unannotated', 'reviewed', 'to_review']);

const parseAnnotationStatus = (value: string | null): DatasetItemAnnotationStatus | null => {
    if (value !== null && VALID_ANNOTATION_STATUSES.has(value as DatasetItemAnnotationStatus)) {
        return value as DatasetItemAnnotationStatus;
    }

    return null;
};

// The `decodeFromBinary` and `encodeToBinary` functions are taken from,
// https://tanstack.com/router/v1/docs/guide/custom-search-param-serialization#using-zipson
// These functions make sure that the strings generated from `stringify` are properly encoded,
// even when `atob` or `btoa` do not guarantee to work with UTF8 characters
const decodeFromBinary = (str: string): string => {
    return decodeURIComponent(
        Array.prototype.map
            .call(atob(str), function (c) {
                return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
            })
            .join('')
    );
};

const encodeToBinary = (str: string): string => {
    return btoa(
        encodeURIComponent(str).replace(/%([0-9A-F]{2})/g, function (_match, p1) {
            return String.fromCharCode(parseInt(p1, 16));
        })
    );
};

export const encodeFilterSearchParam = <T>(filterSearch: T): string => {
    return encodeToBinary(encodeURIComponent(stringify(filterSearch)));
};

const getFilterParam = <T>(filterParam: string): T => {
    try {
        // This may fail if the user manually changes the filter parameter in the url,
        // in that case we ignore the filter
        return (parse(decodeURIComponent(decodeFromBinary(filterParam ?? ''))) ?? {}) as T;
    } catch {
        return {} as T;
    }
};

export const useDatasetFiltersSearchParams = () => {
    const [searchParams, setSearchParams] = useSearchParams();

    const labelsParam = searchParams.get(LABELS_PARAM);
    const filterParam = getFilterParam<string>(labelsParam ?? '');
    const selectedLabelIds = isNonEmptyString(filterParam) ? filterParam.split(',') : [];
    const annotationStatus = parseAnnotationStatus(searchParams.get(ANNOTATION_STATUS_PARAM));
    const startDate = searchParams.get(START_DATE_PARAM);
    const endDate = searchParams.get(END_DATE_PARAM);

    const setSelectedLabelIds = (ids: string[]) => {
        setSearchParams((prev) => {
            if (isEmpty(ids)) {
                prev.delete(LABELS_PARAM);
            } else {
                prev.set(LABELS_PARAM, encodeFilterSearchParam(ids.join(',')));
            }

            return prev;
        });
    };

    const setAnnotationStatus = (status: FilterByStatusKey | null) => {
        setSearchParams((prev) => {
            if (isNil(status)) {
                prev.delete(ANNOTATION_STATUS_PARAM);
            } else {
                prev.set(ANNOTATION_STATUS_PARAM, status);
            }

            return prev;
        });
    };

    const setStartDate = (date: string | null) => {
        setSearchParams((prev) => {
            if (isNil(date)) {
                prev.delete(START_DATE_PARAM);
            } else {
                prev.set(START_DATE_PARAM, date);
            }

            return prev;
        });
    };

    const setEndDate = (date: string | null) => {
        setSearchParams((prev) => {
            if (isNil(date)) {
                prev.delete(END_DATE_PARAM);
            } else {
                prev.set(END_DATE_PARAM, date);
            }

            return prev;
        });
    };

    return {
        selectedLabelIds,
        setSelectedLabelIds,
        annotationStatus,
        setAnnotationStatus,
        startDate,
        setStartDate,
        endDate,
        setEndDate,
    };
};
