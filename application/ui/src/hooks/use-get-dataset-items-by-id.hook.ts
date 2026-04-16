// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo, useRef } from 'react';

import type { DatasetItemAnnotationStatus } from '../constants/shared-types';
import { useGetDatasetItems } from './use-get-dataset-items.hook';

type UseGetDatasetItemsByIdOptions = {
    annotationStatus?: DatasetItemAnnotationStatus;
};

export const useGetDatasetItemsById = ({ annotationStatus }: UseGetDatasetItemsByIdOptions) => {
    const { items, ...response } = useGetDatasetItems({ annotationStatus });

    const accumulatedReviewStatusRef = useRef(new Map<string, boolean>());

    const reviewStatus = useMemo(() => {
        items.forEach(({ id, user_reviewed }) => {
            accumulatedReviewStatusRef.current.set(id, user_reviewed);
        });

        return new Map(accumulatedReviewStatusRef.current);
    }, [items]);

    return { reviewStatus, ...response };
};
