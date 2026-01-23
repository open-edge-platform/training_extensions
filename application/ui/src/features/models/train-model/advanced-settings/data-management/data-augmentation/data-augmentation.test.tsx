// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedConfigurationParameter, getMockedTrainingConfiguration } from 'mocks/mock-training-configuration';
import { render } from 'test-utils/render';

import { TrainingConfiguration } from '../../../../configuration.interface';
import { isBoolParameter } from '../../utils';
import { DataAugmentation } from './data-augmentation.component';

type DataAugmentationParameters = TrainingConfiguration['dataset_preparation']['augmentation'];

const getToggleEnableParameter = (name: string) => {
    return screen.getByRole('switch', { name: `Toggle ${name}` });
};

const toggleParameter = async (name: string) => {
    await userEvent.click(getToggleEnableParameter(name));
};

const getParameter = (name: string) => {
    return screen.getByRole('textbox', { name: `Change ${name}` });
};

describe('DataAugmentation', () => {
    const dataAugmentationParameters = {
        center_crop: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable center crop',
                value: true,
                description: 'Whether to apply center cropping to the image',
                default_value: true,
            }),
            getMockedConfigurationParameter({
                key: 'ratio',
                type: 'float',
                name: 'Crop ratio',
                value: 0.6,
                description: 'Ratio of original dimensions to keep when cropping',
                default_value: 1,
                max_value: null,
                min_value: 0,
            }),
        ],
        random_affine: [
            getMockedConfigurationParameter({
                key: 'enable',
                type: 'bool',
                name: 'Enable random affine',
                value: true,
                description: 'Whether to apply random affine transformations to the image',
                default_value: true,
            }),
            getMockedConfigurationParameter({
                key: 'degrees',
                type: 'float',
                name: 'Rotation degrees',
                value: 15,
                description: 'Maximum rotation angle in degrees',
                default_value: 0,
                max_value: null,
                min_value: 0,
            }),
            getMockedConfigurationParameter({
                key: 'translate_x',
                type: 'float',
                name: 'Horizontal translation',
                value: 0,
                description: 'Maximum horizontal translation as a fraction of image width',
                default_value: 0,
                max_value: null,
                min_value: 0,
            }),
            getMockedConfigurationParameter({
                key: 'translate_y',
                type: 'float',
                name: 'Vertical translation',
                value: 0,
                description: 'Maximum vertical translation as a fraction of image height',
                default_value: 0,
                max_value: null,
                min_value: 0,
            }),
            getMockedConfigurationParameter({
                key: 'scale',
                type: 'float',
                name: 'Scale factor',
                value: 1,
                description: 'Scaling factor for the image during affine transformation',
                default_value: 1,
                max_value: null,
                min_value: 0,
            }),
        ],
    } satisfies DataAugmentationParameters;

    const App = (props: { dataAugmentationParameters: DataAugmentationParameters }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>(() =>
            getMockedTrainingConfiguration({
                dataset_preparation: {
                    filtering: {},
                    subset_split: [],
                    augmentation: props.dataAugmentationParameters,
                },
            })
        );

        const handleUpdateTrainingConfiguration = (
            updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
        ) => {
            setTrainingConfiguration(updateFunction);
        };

        return (
            <DataAugmentation
                parameters={trainingConfiguration?.dataset_preparation.augmentation ?? props.dataAugmentationParameters}
                onUpdateTrainingConfiguration={handleUpdateTrainingConfiguration}
            />
        );
    };

    it('tag displays "Yes" when at least one data augmentation parameter is enabled, "No" otherwise', async () => {
        render(<App dataAugmentationParameters={dataAugmentationParameters} />);

        const enableCenterCrop = dataAugmentationParameters.center_crop[0];
        const enableRandomAffine = dataAugmentationParameters.random_affine[0];

        expect(getToggleEnableParameter(enableCenterCrop.name)).toBeChecked();
        expect(screen.getByLabelText('Data augmentation tag')).toHaveTextContent('Yes');

        await toggleParameter(enableCenterCrop.name);
        await toggleParameter(enableRandomAffine.name);

        expect(getToggleEnableParameter(enableCenterCrop.name)).not.toBeChecked();
        expect(getToggleEnableParameter(enableRandomAffine.name)).not.toBeChecked();
        expect(screen.getByLabelText('Data augmentation tag')).toHaveTextContent('No');
    });

    it('disables all data augmentation parameters when the "Enable" parameter is false, otherwise are enabled', async () => {
        render(<App dataAugmentationParameters={dataAugmentationParameters} />);

        const parameters = Object.values(dataAugmentationParameters);

        for (const parametersGroup of parameters) {
            const enableParameter = parametersGroup[0];
            expect(getToggleEnableParameter(enableParameter.name)).toBeChecked();

            const restOfParameters = parametersGroup.slice(1);

            restOfParameters.forEach((parameter) => {
                expect(getParameter(parameter.name)).toBeEnabled();
            });

            await toggleParameter(enableParameter.name);
            expect(getToggleEnableParameter(enableParameter.name)).not.toBeChecked();

            restOfParameters.forEach((parameter) => {
                expect(getParameter(parameter.name)).toBeDisabled();
            });
        }
    });

    it('updates parameters and resets them to default properly', async () => {
        render(<App dataAugmentationParameters={dataAugmentationParameters} />);

        const parameters = Object.values(dataAugmentationParameters).flat();

        for (const parameter of parameters) {
            if (isBoolParameter(parameter)) {
                expect(getToggleEnableParameter(parameter.name)).toBeChecked();

                await toggleParameter(parameter.name);

                expect(getToggleEnableParameter(parameter.name)).not.toBeChecked();

                await userEvent.click(screen.getByRole('button', { name: `Reset ${parameter.name}` }));
                expect(getToggleEnableParameter(parameter.name)).toBeChecked();
            } else {
                expect(getParameter(parameter.name)).toHaveValue(parameter.value.toString());

                await userEvent.click(screen.getByRole('button', { name: `Increase Change ${parameter.name}` }));

                expect(getParameter(parameter.name)).toHaveValue((Number(parameter.value) + 0.1).toString());

                await userEvent.click(screen.getByRole('button', { name: `Reset ${parameter.name}` }));

                expect(getParameter(parameter.name)).toHaveValue(parameter.default_value.toString());
            }
        }
    });
});
