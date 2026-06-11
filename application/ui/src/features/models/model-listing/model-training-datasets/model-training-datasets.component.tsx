// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, MediaViewModes, Text, ViewModes } from '@geti/ui';
import { useNumberFormatter } from 'react-aria';

import type { DatasetRevision, DatasetSubset } from '../../../../constants/shared-types';
import { useGetDatasetRevisionItems } from '../../../../hooks/use-get-dataset-revision-items.hook';
import { useViewMode } from '../../../../hooks/use-view-mode.hook';
import { GALLERY_VIEW_MODES, type GalleryViewMode } from '../../../../shared/gallery-view-modes';
import { Box } from '../components/box/box.component';
import { SubsetGallery } from './subset-gallery.component';

type SubsetBoxProps = {
    title: string;
    subset: DatasetSubset;
    datasetRevisionId: string;
    totalItems: number;
};

const SubsetBox = ({ title, subset, datasetRevisionId, totalItems }: SubsetBoxProps) => {
    const { items, fetchNextPage, hasNextPage, isFetchingNextPage, isPending, totalCount } = useGetDatasetRevisionItems(
        {
            datasetRevisionId,
            subset,
        }
    );
    const [viewMode, setViewMode] = useViewMode(`model-training-datasets-${subset}-view-mode`, ViewModes.MEDIUM);

    const formatter = useNumberFormatter({ style: 'percent', maximumFractionDigits: 0 });
    const subsetPercentage = totalItems > 0 ? totalCount / totalItems : 0;

    return (
        <Box
            title={`${title} ${formatter.format(subsetPercentage)} (${totalCount})`}
            actions={<MediaViewModes viewMode={viewMode} setViewMode={setViewMode} items={GALLERY_VIEW_MODES} />}
            content={
                <SubsetGallery
                    items={items}
                    datasetRevisionId={datasetRevisionId}
                    viewMode={viewMode as GalleryViewMode}
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
