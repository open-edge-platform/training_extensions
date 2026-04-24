// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@geti/ui';
import { useDatasetMediaWithReviewStatus } from 'hooks/use-dataset-media-with-review-status.hook';

interface TotalItemsProps {
    totalSelectedElements: number;
}
export const TotalItems = ({ totalSelectedElements }: TotalItemsProps) => {
    const hasSelectedElements = totalSelectedElements > 0;
    const { totalCount } = useDatasetMediaWithReviewStatus();

    if (hasSelectedElements) {
        return <Text>{`${totalSelectedElements} selected`}</Text>;
    }

    return <Text>{`${totalCount} media${totalCount === 1 ? '' : 's'}`}</Text>;
};
