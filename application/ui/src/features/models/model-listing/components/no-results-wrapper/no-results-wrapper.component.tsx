// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti/ui';

type NoResultsWrapperProps = {
    message: string;
};

export const NoResultsWrapper = ({ message }: NoResultsWrapperProps) => {
    return (
        <Flex height={'size-1600'} alignItems={'center'} justifyContent={'center'}>
            <Text>{message}</Text>
        </Flex>
    );
};
