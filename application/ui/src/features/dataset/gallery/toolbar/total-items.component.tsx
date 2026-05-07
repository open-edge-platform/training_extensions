// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@geti/ui';
import { useDatasetMediaWithReviewStatus } from 'hooks/use-dataset-media-with-review-status.hook';

type TotalItemsProps = {
    totalSelectedElements: number;
};

export const TotalItems = ({ totalSelectedElements }: TotalItemsProps) => {
    const { totalCount } = useDatasetMediaWithReviewStatus();

    const hasSelectedElements = totalSelectedElements > 0;

    if (hasSelectedElements) {
        return <Text>{`${totalSelectedElements} selected`}</Text>;
    }

    return <Text>{`${totalCount} media item${totalCount === 1 ? '' : 's'}`}</Text>;
};
