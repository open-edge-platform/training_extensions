// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Text } from '@geti/ui';
import { useDatasetMediaWithReviewStatus } from 'hooks/use-dataset-media-with-review-status.hook';

type TotalItemsProps = {
    totalSelectedElements: number;
};

const pluralRules = new Intl.PluralRules('en');

export const TotalItems = ({ totalSelectedElements }: TotalItemsProps) => {
    const { totalCount } = useDatasetMediaWithReviewStatus();

    if (totalCount === 0) {
        return null;
    }

    const hasSelectedElements = totalSelectedElements > 0;

    return (
        <Flex gap={'size-100'}>
            {hasSelectedElements && (
                <>
                    <Text>{`${totalSelectedElements} selected`}</Text>
                    <Divider orientation={'vertical'} size={'S'} />
                </>
            )}

            <Text>{`${totalCount} media ${pluralRules.select(totalCount) === 'one' ? 'item' : 'items'}`}</Text>
        </Flex>
    );
};
