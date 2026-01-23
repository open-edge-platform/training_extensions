// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedConfigurationParameter, getMockedTrainingConfiguration } from 'mocks/mock-training-configuration';
import { render } from 'test-utils/render';

import { TrainingConfiguration } from '../../../../configuration.interface';
import { TILING_MODES } from './tiling-modes.component';
import { Tiling } from './tiling.component';
import { getTilingMode, TILING_AUTOMATIC_DESCRIPTION, TILING_OFF_DESCRIPTION } from './utils';

type TilingParameters = TrainingConfiguration['dataset_preparation']['augmentation']['tiling'];

describe('getTilingMode', () => {
    it('tiling is off when when enable tiling parameter is false', () => {
        expect(
            getTilingMode([
                getMockedConfigurationParameter({
                    key: 'enable',
                    type: 'bool',
                    name: 'Enable tiling',
                    value: false,
                    description: 'Whether to apply tiling to the image',
                    default_value: false,
                }),
            ])
        ).toBe(TILING_MODES.OFF);
    });

    it('tiling is automatic when adaptive tiling parameter is true', () => {
        expect(
            getTilingMode([
                getMockedConfigurationParameter({
                    key: 'enable',
                    type: 'bool',
                    name: 'Enable tiling',
                    value: true,
                    description: 'Whether to apply tiling to the image',
                    default_value: false,
                }),
                getMockedConfigurationParameter({
                    key: 'adaptive_tiling',
                    type: 'bool',
                    name: 'Adaptive tiling',
                    value: true,
                    description: 'Whether to use adaptive tiling based on image content',
                    default_value: false,
                }),
            ])
        ).toBe(TILING_MODES.AUTOMATIC);
    });

    it('tiling is custom when adaptive tiling parameter is false and enable tiling parameter is true', () => {
        expect(
            getTilingMode([
                getMockedConfigurationParameter({
                    key: 'enable',
                    type: 'bool',
                    name: 'Enable tiling',
                    value: true,
                    description: 'Whether to apply tiling to the image',
                    default_value: false,
                }),
                getMockedConfigurationParameter({
                    key: 'adaptive_tiling',
                    type: 'bool',
                    name: 'Adaptive tiling',
                    value: false,
                    description: 'Whether to use adaptive tiling based on image content',
                    default_value: false,
                }),
            ])
        ).toBe(TILING_MODES.CUSTOM);
    });
});

const getTilingModeButton = (tilingMode: TILING_MODES) => {
    return screen.getByRole('button', { name: new RegExp(tilingMode, 'i') });
};

describe('Tiling', () => {
    const customParameters = [
        getMockedConfigurationParameter({
            key: 'tile_size',
            type: 'int',
            name: 'Tile size',
            value: 256,
            description: 'Size of each tile in pixels',
            default_value: 128,
            max_value: null,
            min_value: 0,
        }),
        getMockedConfigurationParameter({
            key: 'tile_overlap',
            type: 'int',
            name: 'Tile overlap',
            value: 64,
            description: 'Overlap between adjacent tiles in pixels',
            default_value: 64,
            max_value: null,
            min_value: 0,
        }),
    ];

    const tilingParameters: TilingParameters = [
        getMockedConfigurationParameter({
            key: 'enable',
            type: 'bool',
            name: 'Enable tiling',
            value: true,
            description: 'Whether to apply tiling to the image',
            default_value: false,
        }),
        getMockedConfigurationParameter({
            key: 'adaptive_tiling',
            type: 'bool',
            name: 'Adaptive tiling',
            value: false,
            description: 'Whether to use adaptive tiling based on image content',
            default_value: false,
        }),
        ...customParameters,
    ];

    const App = (props: { tilingParameters: TilingParameters }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>(() =>
            getMockedTrainingConfiguration({
                dataset_preparation: {
                    filtering: {},
                    subset_split: [],
                    augmentation: {
                        tiling: props.tilingParameters,
                    },
                },
            })
        );

        const handleUpdateTrainingConfiguration = (
            updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
        ) => {
            setTrainingConfiguration(updateFunction);
        };

        return (
            <Tiling
                tilingParameters={
                    trainingConfiguration?.dataset_preparation.augmentation.tiling ?? props.tilingParameters
                }
                onUpdateTrainingConfiguration={handleUpdateTrainingConfiguration}
            />
        );
    };

    it('tiling off and automatic displays only description and no parameters', async () => {
        render(<App tilingParameters={tilingParameters} />);

        await userEvent.click(getTilingModeButton(TILING_MODES.OFF));

        expect(getTilingModeButton(TILING_MODES.OFF)).toHaveAttribute('aria-pressed', 'true');
        expect(getTilingModeButton(TILING_MODES.AUTOMATIC)).toHaveAttribute('aria-pressed', 'false');
        expect(getTilingModeButton(TILING_MODES.CUSTOM)).toHaveAttribute('aria-pressed', 'false');
        expect(screen.getByText(TILING_OFF_DESCRIPTION)).toBeInTheDocument();

        await userEvent.click(getTilingModeButton(TILING_MODES.AUTOMATIC));

        expect(getTilingModeButton(TILING_MODES.AUTOMATIC)).toHaveAttribute('aria-pressed', 'true');
        expect(getTilingModeButton(TILING_MODES.OFF)).toHaveAttribute('aria-pressed', 'false');
        expect(getTilingModeButton(TILING_MODES.CUSTOM)).toHaveAttribute('aria-pressed', 'false');
        expect(screen.getByText(TILING_AUTOMATIC_DESCRIPTION)).toBeInTheDocument();
    });

    it('tiling tag updates properly when tiling mode changes', async () => {
        render(<App tilingParameters={tilingParameters} />);

        await userEvent.click(getTilingModeButton(TILING_MODES.CUSTOM));

        expect(screen.getByLabelText('Tiling tag')).toHaveTextContent(TILING_MODES.CUSTOM);

        await userEvent.click(getTilingModeButton(TILING_MODES.OFF));

        expect(screen.getByLabelText('Tiling tag')).toHaveTextContent(TILING_MODES.OFF);

        await userEvent.click(getTilingModeButton(TILING_MODES.AUTOMATIC));

        expect(screen.getByLabelText('Tiling tag')).toHaveTextContent(TILING_MODES.AUTOMATIC);
    });

    it('custom tiling parameters updates and resets properly', async () => {
        render(<App tilingParameters={tilingParameters} />);

        await userEvent.click(getTilingModeButton(TILING_MODES.CUSTOM));

        expect(getTilingModeButton(TILING_MODES.CUSTOM)).toHaveAttribute('aria-pressed', 'true');
        expect(getTilingModeButton(TILING_MODES.AUTOMATIC)).toHaveAttribute('aria-pressed', 'false');
        expect(getTilingModeButton(TILING_MODES.OFF)).toHaveAttribute('aria-pressed', 'false');

        for (const parameter of customParameters) {
            expect(screen.getByRole('textbox', { name: `Change ${parameter.name}` })).toHaveValue(
                parameter.value.toString()
            );
            await userEvent.click(screen.getByRole('button', { name: `Increase Change ${parameter.name}` }));
            expect(screen.getByRole('textbox', { name: `Change ${parameter.name}` })).toHaveValue(
                (Number(parameter.value) + 1).toString()
            );

            await userEvent.click(screen.getByRole('button', { name: `Reset ${parameter.name}` }));
            expect(screen.getByRole('textbox', { name: `Change ${parameter.name}` })).toHaveValue(
                parameter.default_value.toString()
            );
        }
    });
});
