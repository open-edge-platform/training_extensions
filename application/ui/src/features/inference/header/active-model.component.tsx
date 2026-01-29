// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useGetActiveModelArchitectureId } from '../../models/hooks/api/use-get-active-model-architecture-id.hook';

export const ActiveModel = () => {
    const activeModelArchitectureId = useGetActiveModelArchitectureId();

    return (
        <Flex gap='size-50' alignItems='center'>
            <Text
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-900)',
                }}
            >
                Model:
            </Text>
            <Text
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-700)',
                }}
            >
                {isEmpty(activeModelArchitectureId) ? 'Unknown' : activeModelArchitectureId}
            </Text>
        </Flex>
    );
};
