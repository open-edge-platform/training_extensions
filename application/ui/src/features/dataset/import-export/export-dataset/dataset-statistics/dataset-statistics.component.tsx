// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Text, View } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../../api/client';

import classes from './dataset-statistics.module.scss';

export const DatasetStatistics = () => {
    const projectId = useProjectIdentifier();

    const { data: annotatedItems } = $api.useQuery('get', '/api/projects/{project_id}/dataset/items', {
        params: {
            path: { project_id: projectId },
            query: { limit: 1, annotation_status: 'reviewed' },
        },
    });

    const { data: mediaItems } = $api.useQuery('get', '/api/projects/{project_id}/dataset/items', {
        params: { path: { project_id: projectId } },
    });

    const totalMediaItems = mediaItems?.pagination.total ?? 0;
    const totalAnnotatedItems = annotatedItems?.pagination.total ?? 0;
    const totalUnannotatedItems = totalMediaItems - totalAnnotatedItems;
    const percentageAnnotated = totalMediaItems > 0 ? Math.round((totalAnnotatedItems / totalMediaItems) * 100) : 0;
    const percentageUnannotated = totalMediaItems > 0 ? Math.round((totalUnannotatedItems / totalMediaItems) * 100) : 0;

    return (
        <View backgroundColor='gray-75' padding='size-200' borderRadius='regular'>
            <Flex alignItems='center' justifyContent='center' gap='size-200'>
                <Flex
                    alignSelf='start'
                    direction='column'
                    alignItems='end'
                    justifyContent='center'
                    UNSAFE_className={classes.unannotatedStats}
                >
                    <Text>Unannotated</Text>
                    <Text>{percentageUnannotated}%</Text>
                    <Text>{totalUnannotatedItems} images</Text>
                </Flex>

                <Flex
                    width='size-1600'
                    height='size-1600'
                    direction='column'
                    alignItems='center'
                    justifyContent='center'
                    UNSAFE_className={classes.donut}
                    UNSAFE_style={{ '--percentage': `${percentageAnnotated}%` }}
                >
                    <Flex direction='column'>
                        <Text UNSAFE_className={classes.totalMediaItems}>{totalMediaItems}</Text>
                        <Text UNSAFE_className={classes.mediaSubtitle}>Images</Text>
                    </Flex>
                </Flex>

                <Flex
                    alignSelf='end'
                    direction='column'
                    alignItems='start'
                    justifyContent='center'
                    UNSAFE_className={classes.unannotatedStats}
                >
                    <Text>Annotated</Text>
                    <Text>{percentageAnnotated}%</Text>
                    <Text>{totalAnnotatedItems} images</Text>
                </Flex>
            </Flex>
        </View>
    );
};
