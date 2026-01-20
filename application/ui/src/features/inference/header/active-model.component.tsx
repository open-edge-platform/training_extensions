// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';

import { $api } from '../../../api/client';

export const ActiveModel = () => {
    const projectId = useProjectIdentifier();

    const modelsQuery = $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
        params: { path: { project_id: projectId } },
    });

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
                {!isEmpty(modelsQuery.data) ? modelsQuery.data[0].architecture : 'Unknown'}
            </Text>
        </Flex>
    );
};
