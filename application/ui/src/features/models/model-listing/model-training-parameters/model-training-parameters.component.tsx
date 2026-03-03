// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid, Text } from '@geti/ui';

import { useGetModelTrainingConfiguration } from '../../hooks/api/use-get-model-training-configuration.hook';
import { Box } from '../components/box/box.component';

type ModelTrainingParametersProps = {
    modelId: string;
};

export const ModelTrainingParameters = ({ modelId }: ModelTrainingParametersProps) => {
    const { data } = useGetModelTrainingConfiguration(modelId);

    return (
        <Grid columns={['1fr', '1fr', '1fr', '1fr']} gap={'size-200'}>
            <Box
                title={'LEARNING PARAMETERS'}
                content={
                    <Grid columns={['1fr', '1fr']} gap={'size-100'}>
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
                    </Grid>
                }
            />
            <Box
                title={'FILTERS'}
                content={
                    <Grid columns={['1fr', '1fr']} gap={'size-100'}>
                        <Text>Param 1:</Text>
                        <Text>Value 1</Text>

                        <Text>Param 2:</Text>
                        <Text>Value 2</Text>

                        <Text>Param 3:</Text>
                        <Text>Value 3</Text>

                        <Text>Param 4:</Text>
                        <Text>Value 4</Text>
                    </Grid>
                }
            />
            <Box
                title={'FINE-TUNE'}
                content={
                    <Grid columns={['1fr', '1fr']} gap={'size-100'}>
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
                    </Grid>
                }
            />
            <Box
                title={'TILING'}
                content={
                    <Grid columns={['1fr', '1fr']} gap={'size-100'}>
                        <Text>Off</Text>
                    </Grid>
                }
            />
        </Grid>
    );
};
