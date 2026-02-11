// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Text, View } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';

export const DatasetStatistics = () => {
    const project_id = useProjectIdentifier();

    const { data: datasetItems } = $api.useQuery('get', '/api/projects/{project_id}/dataset/items', {
        params: { path: { project_id } },
    });
    const totalItems = datasetItems?.pagination.total ?? 0;

    return (
        <View backgroundColor='gray-75' padding='size-200' borderRadius='regular'>
            <Flex direction='column' alignItems='center' justifyContent='center'>
                <Text UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>{totalItems}</Text>
                <Text UNSAFE_style={{ fontSize: dimensionValue('size-175'), fontWeight: '500' }}>Images</Text>
            </Flex>
        </View>
    );
};
