// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, dimensionValue, Flex, Grid, Meter, Text } from '@geti-ui/ui';
import { useDatasetStatistics } from 'hooks/api/dataset.hook';
import { isEmpty } from 'lodash-es';

import { DatasetCard } from './dataset-card.component';
import { DatasetLabelsChart } from './dataset-labels-chart.component';

import classes from './dataset-statistics.module.scss';

const getAnnotationPercentage = (annotated: number, total: number): number => {
    if (total === 0) {
        return 0;
    }
    const percentage = (annotated / total) * 100;
    return Math.min(100, Math.max(0, percentage));
};

const getMaxInstancesPerLabel = (instancesPerLabel: { label_id: string | null; instances: number }[]): number => {
    if (isEmpty(instancesPerLabel)) {
        return 0;
    }

    return instancesPerLabel.reduce((max, { instances }) => (instances > max ? instances : max), 0);
};

export const DatasetStatisticsContent = () => {
    const { data: statistics } = useDatasetStatistics();

    const totalMediaItems = statistics.media_counts.images + statistics.media_counts.videos;

    const maxInstancesPerLabel = getMaxInstancesPerLabel(statistics.annotations_counts.instances_per_label);

    const totalItems = Math.max(totalMediaItems, maxInstancesPerLabel);

    const annotationPercentage = getAnnotationPercentage(
        statistics.annotations_counts.annotated_images,
        statistics.media_counts.images
    );

    return (
        <Content>
            <Grid
                gap='size-200'
                rows={['auto', 'auto']}
                areas={['col1 col2 col3', 'full full full']}
                columns={['1fr', '1fr', '1fr']}
                UNSAFE_style={{
                    padding: dimensionValue('size-300'),
                    background: 'var(--spectrum-global-color-gray-50)',
                }}
            >
                <DatasetCard title='Number of media' gridArea='col1'>
                    <Flex justifyContent={'space-evenly'}>
                        <Flex direction={'column'} alignItems={'center'}>
                            <Text UNSAFE_className={classes.mainValue}>{statistics.media_counts.images}</Text>
                            <Text UNSAFE_className={classes.subTitle}>Images</Text>
                        </Flex>

                        <Flex direction={'column'} alignItems={'center'}>
                            <Text UNSAFE_className={classes.mainValue}>{statistics.media_counts.videos}</Text>
                            <Text UNSAFE_className={classes.subTitle}>Videos</Text>
                        </Flex>
                    </Flex>
                </DatasetCard>

                <DatasetCard title='Annotated images' gridArea='col2'>
                    <Flex direction={'column'} alignItems={'center'}>
                        <Text UNSAFE_className={classes.mainValue}>
                            {statistics.annotations_counts.annotated_images}
                        </Text>

                        <Meter
                            label=' '
                            value={annotationPercentage}
                            variant='positive'
                            labelPosition='side'
                            UNSAFE_className={classes.meter}
                        />
                    </Flex>
                </DatasetCard>
                <DatasetCard title='Annotated videos / frames' gridArea='col3'>
                    <Flex gap={'size-125'} alignItems={'center'}>
                        <Text UNSAFE_className={classes.subTitle}>Videos:</Text>
                        <Text UNSAFE_className={classes.secondaryValue}>
                            {statistics.annotations_counts.annotated_videos}
                        </Text>
                    </Flex>

                    <Flex gap={'size-125'} alignItems={'center'}>
                        <Text UNSAFE_className={classes.subTitle}>Frames:</Text>
                        <Text UNSAFE_className={classes.secondaryValue}>
                            {statistics.annotations_counts.annotated_video_frames}
                        </Text>
                    </Flex>
                </DatasetCard>
                <DatasetCard title='Number of objects per label' gridArea='full' hasFullSizeContent>
                    <DatasetLabelsChart
                        totalItems={totalItems}
                        instancesPerLabel={statistics.annotations_counts.instances_per_label}
                    />
                </DatasetCard>
            </Grid>
        </Content>
    );
};
