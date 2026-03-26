// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    dimensionValue,
    Divider,
    Flex,
    Grid,
    Heading,
    Meter,
    Text,
} from '@geti/ui';
import { GraphChart } from '@geti/ui/icons';
import { useDatasetStatistics } from 'hooks/api/dataset.hook';

import classes from './dataset-statistics.module.scss';

type CardProps = {
    title: string;
    gridArea: string;
    children: ReactNode;
};

const Card = ({ title, gridArea, children }: CardProps) => {
    return (
        <Flex
            gap={'size-100'}
            width={'100%'}
            gridArea={gridArea}
            direction={'column'}
            UNSAFE_style={{ padding: dimensionValue('size-200'), background: 'var(--spectrum-global-color-gray-100)' }}
        >
            <Heading>{title}</Heading>
            <Divider size='S' />
            {children}
        </Flex>
    );
};

export const DatasetStatistics = () => {
    const { data: statistics } = useDatasetStatistics();

    const annotationPercentage =
        (statistics.annotations_counts.annotated_images / statistics.media_counts.images) * 100;

    return (
        <DialogTrigger>
            <ActionButton isQuiet aria-label={'dataset statistics'}>
                <GraphChart />
            </ActionButton>
            {(close) => (
                <Dialog width={{ base: '90vw', L: '70vw' }}>
                    <Heading>Dataset Statistics</Heading>
                    <Divider />
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
                            <Card title='Number of media' gridArea='col1'>
                                <Flex justifyContent={'space-evenly'}>
                                    <Flex direction={'column'} alignItems={'center'}>
                                        <Text UNSAFE_className={classes.mainValue}>
                                            {statistics.media_counts.images}
                                        </Text>
                                        <Text UNSAFE_className={classes.subTitle}>Images</Text>
                                    </Flex>

                                    <Flex direction={'column'} alignItems={'center'}>
                                        <Text UNSAFE_className={classes.mainValue}>
                                            {statistics.media_counts.videos}
                                        </Text>
                                        <Text UNSAFE_className={classes.subTitle}>Videos</Text>
                                    </Flex>
                                </Flex>
                            </Card>

                            <Card title='Annotated images' gridArea='col2'>
                                <Flex direction={'column'} alignItems={'center'}>
                                    <Text UNSAFE_className={classes.mainValue}>
                                        {statistics.annotations_counts.annotated_images}
                                    </Text>

                                    <Meter
                                        label='-'
                                        value={annotationPercentage}
                                        variant='positive'
                                        labelPosition='side'
                                        UNSAFE_className={classes.meter}
                                    />
                                </Flex>
                            </Card>
                            <Card title='Annotated videos / frames' gridArea='col3'>
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
                            </Card>
                            <Card title='Number of objects per label' gridArea='full'>
                                <Text>Number of objects per label</Text>
                            </Card>
                        </Grid>
                    </Content>
                    <ButtonGroup>
                        <Button variant='secondary' onPress={close}>
                            Close
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogTrigger>
    );
};
