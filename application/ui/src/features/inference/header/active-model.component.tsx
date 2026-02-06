// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti/ui';
import { isEmpty } from 'lodash-es';

import { useGetActiveModel } from '../../models/hooks/api/use-get-active-model.hook';

export const ActiveModel = () => {
    const activeModel = useGetActiveModel();

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
                {isEmpty(activeModel) ? 'Unknown' : activeModel.name}
            </Text>
        </Flex>
    );
};
