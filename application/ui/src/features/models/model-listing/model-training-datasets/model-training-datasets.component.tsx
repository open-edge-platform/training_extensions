// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Flex, Text } from '@geti/ui';
import { Filter, GridSmall, Search, SortUpDown } from '@geti/ui/icons';
import { useNumberFormatter } from 'react-aria';

import type { DatasetRevision, DatasetSubset } from '../../../../constants/shared-types';
import { useGetDatasetRevisionItems } from '../../../../hooks/use-get-dataset-revision-items.hook';
import { Box } from '../components/box/box.component';
import { SubsetGallery } from './subset-gallery.component';

import classes from './model-training-datasets.module.scss';

type SubsetBoxProps = {
    title: string;
    subset: DatasetSubset;
    datasetRevisionId: string;
    totalItems: number;
};

// TODO: Uncomment when we want to support subset actions
// eslint-disable-next-line @typescript-eslint/no-unused-vars
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

    const formatter = useNumberFormatter({ style: 'percent', maximumFractionDigits: 0 });
    const subsetPercentage = totalItems > 0 ? totalCount / totalItems : 0;

    return (
        <Box
            customClasses={classes.box}
            headingClassName={classes.boxHeading}
            contentClassName={classes.boxContent}
            title={`${title} ${formatter.format(subsetPercentage)} (${totalCount})`}
            content={
                <SubsetGallery
                    items={items}
                    datasetRevisionId={datasetRevisionId}
                    fetchNextPage={fetchNextPage}
                    hasNextPage={hasNextPage}
                    isFetchingNextPage={isFetchingNextPage}
                    isPending={isPending}
                />
            }
        />
    );
};

const ModelTrainingContent = ({ datasetRevision }: { datasetRevision: DatasetRevision }) => {
    const totalItems = datasetRevision.item_counts?.total ?? 0;
    const datasetRevisionId = String(datasetRevision.id);

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

export const ModelTrainingDatasets = ({ datasetRevision }: { datasetRevision?: DatasetRevision }) => {
    if (!datasetRevision || !datasetRevision.id) {
        return (
            <Flex justifyContent={'center'} alignItems={'center'} height={'size-3000'}>
                <Text>No dataset revision found for this model</Text>
            </Flex>
        );
    }

    if (datasetRevision.files_deleted) {
        return (
            <Flex justifyContent={'center'} alignItems={'center'} height={'size-3000'}>
                <Text>The files for this dataset revision have been deleted.</Text>
            </Flex>
        );
    }

    return <ModelTrainingContent datasetRevision={datasetRevision} />;
};
