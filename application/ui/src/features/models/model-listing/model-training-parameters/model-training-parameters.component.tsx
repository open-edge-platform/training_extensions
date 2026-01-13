// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Content, Flex, Grid, Heading, Text } from '@geti/ui';

import styles from './model-training-parameters.module.scss';

const Box = ({ title, content }: { title: string; content: ReactNode }) => {
    return (
        <Flex direction={'column'} height={'100%'}>
            <Heading UNSAFE_className={styles.boxHeading} level={5}>
                {title}
            </Heading>
            <Content UNSAFE_className={styles.boxContent}>
                <Grid columns={['1fr', '1fr']} gap={'size-100'}>
                    {content}
                </Grid>
            </Content>
        </Flex>
    );
};

export const ModelTrainingParameters = () => {
    return (
        <Grid columns={['1fr', '1fr', '1fr', '1fr']} gap={'size-200'}>
            <Box
                title={'LEARNING PARAMETERS'}
                content={
                    <>
                        <Text>Input size:</Text>
                        <Text>256 x 640 px</Text>

                        <Text>Maximum epochs:</Text>
                        <Text>200</Text>

                        <Text>Row 3:</Text>
                        <Text>Value 3</Text>

                        <Text>Row 4:</Text>
                        <Text>Value 4</Text>

                        <Text>Row 5:</Text>
                        <Text>Value 5</Text>
                    </>
                }
            />
            <Box
                title={'FILTERS'}
                content={
                    <>
                        <Text>Param 1:</Text>
                        <Text>Value 1</Text>

                        <Text>Param 2:</Text>
                        <Text>Value 2</Text>

                        <Text>Param 3:</Text>
                        <Text>Value 3</Text>

                        <Text>Param 4:</Text>
                        <Text>Value 4</Text>
                    </>
                }
            />
            <Box
                title={'FINE-TUNE'}
                content={
                    <>
                        <Text>Param 1:</Text>
                        <Text>Value 1</Text>

                        <Text>Param 2:</Text>
                        <Text>Value 2</Text>

                        <Text>Param 3:</Text>
                        <Text>Value 3</Text>

                        <Text>Param 4:</Text>
                        <Text>Value 4</Text>

                        <Text>Param 5:</Text>
                        <Text>Value 5</Text>
                    </>
                }
            />
            <Box
                title={'TILING'}
                content={
                    <>
                        <Text>Off</Text>
                    </>
                }
            />
        </Grid>
    );
};
