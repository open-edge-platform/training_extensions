// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Grid, minmax, Text, View } from '@geti/ui';

import { ConfigurationParameter, TrainingConfiguration } from '../../../../configuration.interface';
import { Accordion } from '../../ui/accordion/accordion.component';
import { Parameters } from '../../ui/parameters.component';
import { TILING_MODES, TilingModes } from './tiling-modes.component';
import {
    getAdaptiveTilingParameter,
    getCustomTilingParameters,
    getEnableTilingParameter,
    getTilingMode,
    TILING_AUTOMATIC_DESCRIPTION,
    TILING_OFF_DESCRIPTION,
} from './utils';

import styles from './tiling.module.scss';

type TilingProps = {
    tilingParameters: ConfigurationParameter[];
    onUpdateTrainingConfiguration: (
        updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
    ) => void;
};

export const Tiling = ({ tilingParameters, onUpdateTrainingConfiguration }: TilingProps) => {
    const selectedTilingMode = getTilingMode(tilingParameters);

    const customTilingParameters = getCustomTilingParameters(tilingParameters);

    const handleUpdateTilingParameter = (inputParameter: ConfigurationParameter | ConfigurationParameter[]) => {
        onUpdateTrainingConfiguration((config) => {
            if (config === undefined) return;

            const updatedTilingParameters = tilingParameters.map((parameter) => {
                if (Array.isArray(inputParameter)) {
                    const parameterToUpdate = inputParameter.find((p) => p.key === parameter.key);

                    return parameterToUpdate ?? parameter;
                }

                return parameter.key === inputParameter.key ? inputParameter : parameter;
            });

            return {
                ...config,
                datasetPreparation: {
                    ...config.datasetPreparation,
                    augmentation: {
                        ...config.datasetPreparation.augmentation,
                        tiling: updatedTilingParameters,
                    },
                },
            };
        });
    };

    const handleTilingModeChange = (tilingMode: TILING_MODES) => {
        const adaptiveParameter = getAdaptiveTilingParameter(tilingParameters);
        const enableParameter = getEnableTilingParameter(tilingParameters);

        if (adaptiveParameter === undefined || enableParameter === undefined) {
            return;
        }

        if (tilingMode === TILING_MODES.AUTOMATIC) {
            handleUpdateTilingParameter([
                { ...enableParameter, value: true },
                { ...adaptiveParameter, value: true },
            ]);
        } else if (tilingMode === TILING_MODES.OFF) {
            handleUpdateTilingParameter([
                { ...enableParameter, value: false },
                { ...adaptiveParameter, value: false },
            ]);
        } else {
            handleUpdateTilingParameter([
                { ...enableParameter, value: true },
                { ...adaptiveParameter, value: false },
            ]);
        }
    };

    const TILING_MODE_COMPONENTS: Record<TILING_MODES, ReactNode> = {
        [TILING_MODES.OFF]: (
            <Text UNSAFE_className={styles.tilingModeDescription} gridColumn={'2/3'}>
                {TILING_OFF_DESCRIPTION}
            </Text>
        ),

        [TILING_MODES.AUTOMATIC]: (
            <View UNSAFE_className={styles.tilingModeDescription} gridColumn={'2/3'}>
                {TILING_AUTOMATIC_DESCRIPTION}
            </View>
        ),
        [TILING_MODES.CUSTOM]: (
            <View gridColumn={'1/-1'}>
                <Parameters parameters={customTilingParameters} onChange={handleUpdateTilingParameter} />
            </View>
        ),
    };

    return (
        <Accordion>
            <Accordion.Title>
                Tiling<Accordion.Tag ariaLabel={'Tiling tag'}>{selectedTilingMode}</Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>
                    Tiling is a technique that divides high-resolution images into smaller tiles and might be useful to
                    increase accuracy for small object detection tasks.
                </Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <Grid
                    columns={['size-3000', minmax('size-3400', '1fr'), 'size-400']}
                    gap={'size-300'}
                    alignItems={'center'}
                >
                    <TilingModes selectedTilingMode={selectedTilingMode} onTilingModeChange={handleTilingModeChange} />
                    {TILING_MODE_COMPONENTS[selectedTilingMode]}
                </Grid>
            </Accordion.Content>
        </Accordion>
    );
};
