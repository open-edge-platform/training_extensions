// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Content, Flex, Heading, Text } from '@geti/ui';
import { Filter, GridSmall, Search, SortUpDown } from '@geti/ui/icons';

import type { DatasetSubset, Model } from '../../../../constants/shared-types';
import { useGetDatasetRevisionItems } from '../../../../hooks/use-get-dataset-revision-items.hook';
import { SubsetGallery } from './subset-gallery.component';

import styles from './model-training-datasets.module.scss';

type SubsetBoxProps = {
    title: string;
    subsetSplit: number;
    subset: DatasetSubset;
    datasetRevisionId: string;
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

const SubsetBox = ({ title, subsetSplit, subset, datasetRevisionId }: SubsetBoxProps) => {
    const { items, fetchNextPage, hasNextPage, isFetchingNextPage, isPending, totalCount } = useGetDatasetRevisionItems(
        {
            datasetRevisionId,
            subset,
        }
    );

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
                        {title} {subsetSplit}%
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

export const ModelTrainingDatasets = ({ model }: { model: Model }) => {
    const datasetRevisionId = model.training_info.dataset_revision_id;

    if (!datasetRevisionId) {
        return <Text>No dataset revision found for this model</Text>;
    }

    return (
        <Flex gap={'size-300'} width={'100%'}>
            <SubsetBox title={'Training'} subsetSplit={70} subset={'training'} datasetRevisionId={datasetRevisionId} />
            <SubsetBox
                title={'Validation'}
                subsetSplit={20}
                subset={'validation'}
                datasetRevisionId={datasetRevisionId}
            />
            <SubsetBox title={'Testing'} subsetSplit={10} subset={'testing'} datasetRevisionId={datasetRevisionId} />
        </Flex>
    );
};
