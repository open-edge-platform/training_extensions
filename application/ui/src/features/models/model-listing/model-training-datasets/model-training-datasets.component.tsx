// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Content, Flex, Heading, Text } from '@geti/ui';
import { Filter, GridSmall, Search, SortUpDown } from '@geti/ui/icons';

import type { DatasetSubset } from '../../../../constants/shared-types';
import { useGetDatasetRevisionItems } from '../../../../hooks/use-get-dataset-revision-items.hook';
import { SubsetGallery } from './subset-gallery.component';

import styles from './model-training-datasets.module.scss';

type SubsetBoxProps = {
    title: string;
    subset: DatasetSubset;
    datasetRevisionId: string;
    totalItems: number;
};

const BoxActions = () => {
    return (
        <Flex marginStart={'auto'} gap={'size-50'}>
            <ActionButton isQuiet aria-label='Search dataset'>
                <Search />
            </ActionButton>
            <ActionButton isQuiet aria-label='Filter dataset'>
                <Filter />
            </ActionButton>
            <ActionButton isQuiet aria-label='Grid view'>
                <GridSmall />
            </ActionButton>
            <ActionButton isQuiet aria-label='Sort dataset'>
                <SortUpDown />
            </ActionButton>
        </Flex>
    );
};

const SubsetBox = ({ title, subset, datasetRevisionId, totalItems }: SubsetBoxProps) => {
    const { items, fetchNextPage, hasNextPage, isFetchingNextPage, isPending, totalCount } = useGetDatasetRevisionItems(
        {
            datasetRevisionId,
            subset,
        }
    );

    const subsetPercentage = totalItems > 0 ? Math.round((totalCount / totalItems) * 100) : 0;

    return (
        <Flex
            minWidth={0}
            flex={1}
            direction={'column'}
            height={'100%'}
            minHeight={'size-5000'}
            justifyContent={'center'}
        >
            <Flex UNSAFE_className={styles.boxHeading} justifyContent={'space-between'} alignItems={'center'}>
                <Flex gap={'size-100'} alignItems={'center'}>
                    <Heading level={5}>
                        {title} {subsetPercentage}%
                    </Heading>
                    <Text>({totalCount})</Text>
                </Flex>
                <BoxActions />
            </Flex>

            <Content UNSAFE_className={styles.boxContent}>
                <SubsetGallery
                    items={items}
                    datasetRevisionId={datasetRevisionId}
                    fetchNextPage={fetchNextPage}
                    hasNextPage={hasNextPage}
                    isFetchingNextPage={isFetchingNextPage}
                    isLoading={isPending}
                />
            </Content>
        </Flex>
    );
};

export const ModelTrainingDatasets = ({ datasetRevisionId }: { datasetRevisionId: string | undefined | null }) => {
    const { totalCount: trainingCount } = useGetDatasetRevisionItems({
        datasetRevisionId: datasetRevisionId ?? '',
        subset: 'training',
    });
    const { totalCount: validationCount } = useGetDatasetRevisionItems({
        datasetRevisionId: datasetRevisionId ?? '',
        subset: 'validation',
    });
    const { totalCount: testingCount } = useGetDatasetRevisionItems({
        datasetRevisionId: datasetRevisionId ?? '',
        subset: 'testing',
    });

    const totalItems = trainingCount + validationCount + testingCount;

    if (!datasetRevisionId) {
        return <Text>No dataset revision found for this model</Text>;
    }

    return (
        <Flex gap={'size-300'} width={'100%'}>
            <SubsetBox
                title={'Training'}
                subset={'training'}
                datasetRevisionId={datasetRevisionId}
                totalItems={totalItems}
            />
            <SubsetBox
                title={'Validation'}
                subset={'validation'}
                datasetRevisionId={datasetRevisionId}
                totalItems={totalItems}
            />
            <SubsetBox
                title={'Testing'}
                subset={'testing'}
                datasetRevisionId={datasetRevisionId}
                totalItems={totalItems}
            />
        </Flex>
    );
};
