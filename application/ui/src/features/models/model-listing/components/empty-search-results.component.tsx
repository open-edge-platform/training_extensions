// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Heading } from '@geti/ui';
import { Search } from '@geti/ui/icons';

export const EmptySearchResults = () => {
    return (
        <Flex
            direction={'column'}
            alignItems={'center'}
            justifyContent={'center'}
            gap={'size-200'}
            marginY={'size-1300'}
        >
            <Search />
            <Heading level={3}>No models found</Heading>
        </Flex>
    );
};
