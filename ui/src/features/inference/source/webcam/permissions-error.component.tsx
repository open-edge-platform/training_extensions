// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Heading, Text } from '@geti/ui';

export const errorMessage = 'Please check your device and network settings and try again.';

export const PermissionError = () => {
    return (
        <Flex direction={'column'} gap={'size-100'}>
            <Heading level={2} margin={0}>
                Camera connection is lost
            </Heading>
            <Text>{errorMessage}</Text>
        </Flex>
    );
};
