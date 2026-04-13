// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';
import { render } from 'test-utils/render';

import { TrainingConfiguration } from '../../../../../../constants/shared-types';
import { isParameter, isParameterGroup } from '../../../../model-listing/model-training-parameters/utils';
import { isBoolEnableParameter, isBoolParameter } from '../../utils';
import { DataAugmentation } from './data-augmentation.component';
import { DataAugmentationConfigurationParameters, getDataAugmentationParameters } from './utils';

const getToggleEnableParameter = (name: string) => {
    return screen.getByRole('switch', { name: `Toggle ${name}` });
};

const toggleParameter = (name: string) => {
    fireEvent.click(getToggleEnableParameter(name));
};

const getParameter = (name: string) => {
    return screen.getByRole('textbox', { name: `Change ${name}` });
};

describe('DataAugmentation', () => {
    const dataAugmentationParameters = getMockedConfigurationParameterGroup({
        key: 'augmentation',
        name: 'Augmentation',
        parameters: [
            getMockedConfigurationParameter({
                key: 'deim_framework',
                value_type: 'bool',
                name: 'DEIM framework',
                depends_on: null,
                value: false,
            }),
            getMockedConfigurationParameterGroup({
                key: 'center_crop',
                name: 'Center crop',
                depends_on: {
                    deim_framework: [false, null],
                },
                parameters: [
                    getMockedConfigurationParameter({
                        key: 'enable',
                        value_type: 'bool',
                        name: 'Enable center crop',
                        value: true,
                        description: 'Whether to apply center cropping to the image',
                        default_value: true,
                    }),
                    getMockedConfigurationParameter({
                        key: 'ratio',
                        value_type: 'float',
                        name: 'Crop ratio',
                        value: 0.6,
                        description: 'Ratio of original dimensions to keep when cropping',
                        default_value: 1,
                        max_value: null,
                        min_value: 0,
                    }),
                ],
            }),
            getMockedConfigurationParameterGroup({
                key: 'random_affine',
                name: 'Random affine',
                depends_on: {
                    deim_framework: [false, null],
                },
                parameters: [
                    getMockedConfigurationParameter({
                        key: 'enable',
                        value_type: 'bool',
                        name: 'Enable random affine',
                        value: true,
                        description: 'Whether to apply random affine transformations to the image',
                        default_value: true,
                    }),
                    getMockedConfigurationParameter({
                        key: 'degrees',
                        value_type: 'float',
                        name: 'Rotation degrees',
                        value: 15,
                        description: 'Maximum rotation angle in degrees',
                        default_value: 0,
                        max_value: null,
                        min_value: 0,
                    }),
                    getMockedConfigurationParameter({
                        key: 'translate_x',
                        value_type: 'float',
                        name: 'Horizontal translation',
                        value: 0,
                        description: 'Maximum horizontal translation as a fraction of image width',
                        default_value: 0,
                        max_value: null,
                        min_value: 0,
                    }),
                    getMockedConfigurationParameter({
                        key: 'translate_y',
                        value_type: 'float',
                        name: 'Vertical translation',
                        value: 0,
                        description: 'Maximum vertical translation as a fraction of image height',
                        default_value: 0,
                        max_value: null,
                        min_value: 0,
                    }),
                    getMockedConfigurationParameter({
                        key: 'scale',
                        value_type: 'float',
                        name: 'Scale factor',
                        value: 1,
                        description: 'Scaling factor for the image during affine transformation',
                        default_value: 1,
                        max_value: null,
                        min_value: 0,
                    }),
                ],
            }),
        ],
    });

    const App = (props: { dataAugmentationParameters: DataAugmentationConfigurationParameters }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>({
            parameters: [
                getMockedConfigurationParameterGroup({
                    key: 'dataset_preparation',
                    parameters: [dataAugmentationParameters],
                }),
            ],
        });

        return (
            <DataAugmentation
                dataAugmentationParameters={
                    getDataAugmentationParameters(trainingConfiguration ?? { parameters: [] }) ??
                    props.dataAugmentationParameters
                }
                onTrainingConfigurationChange={setTrainingConfiguration}
            />
        );
    };

    it('tag displays "Yes" when at least one data augmentation parameter is enabled, "No" otherwise', () => {
        render(<App dataAugmentationParameters={dataAugmentationParameters} />);

        const centerCropGroup = dataAugmentationParameters.parameters.find(
            (g) => g.key === 'center_crop'
        ) as DataAugmentationConfigurationParameters;
        const randomAffineGroup = dataAugmentationParameters.parameters.find(
            (g) => g.key === 'random_affine'
        ) as DataAugmentationConfigurationParameters;
        const enableCenterCrop = centerCropGroup.parameters[0];
        const enableRandomAffine = randomAffineGroup.parameters[0];

        expect(getToggleEnableParameter(enableCenterCrop.name)).toBeChecked();
        expect(screen.getByLabelText('Data augmentation tag')).toHaveTextContent('Yes');

        toggleParameter(enableCenterCrop.name);
        toggleParameter(enableRandomAffine.name);

        expect(getToggleEnableParameter(enableCenterCrop.name)).not.toBeChecked();
        expect(getToggleEnableParameter(enableRandomAffine.name)).not.toBeChecked();
        expect(screen.getByLabelText('Data augmentation tag')).toHaveTextContent('No');
    });

    it('disables all data augmentation parameters when the "Enable" parameter is false, otherwise are enabled', () => {
        render(<App dataAugmentationParameters={dataAugmentationParameters} />);

        dataAugmentationParameters.parameters.forEach((parametersGroup) => {
            if (isParameterGroup(parametersGroup)) {
                const enableParameter = parametersGroup.parameters[0];
                expect(getToggleEnableParameter(enableParameter.name)).toBeChecked();

                const restOfParameters = parametersGroup.parameters.slice(1);

                restOfParameters.forEach((parameter) => {
                    expect(getParameter(parameter.name)).toBeEnabled();
                });

                toggleParameter(enableParameter.name);
                expect(getToggleEnableParameter(enableParameter.name)).not.toBeChecked();

                restOfParameters.forEach((parameter) => {
                    expect(getParameter(parameter.name)).toBeDisabled();
                });
            }
        });
    });

    it('updates parameters and resets them to default properly', async () => {
        render(<App dataAugmentationParameters={dataAugmentationParameters} />);

        for (const parametersGroup of dataAugmentationParameters.parameters) {
            if (isParameterGroup(parametersGroup)) {
                for (const parameter of parametersGroup.parameters) {
                    if (!isParameter(parameter)) continue;

                    if (isBoolEnableParameter(parameter)) {
                        expect(getToggleEnableParameter(parameter.name)).toBeChecked();

                        toggleParameter(parameter.name);

                        expect(getToggleEnableParameter(parameter.name)).not.toBeChecked();

                        await userEvent.click(screen.getByRole('button', { name: `Reset ${parametersGroup.name}` }));
                        expect(getToggleEnableParameter(parameter.name)).toBeChecked();
                    } else if (isBoolParameter(parameter)) {
                        expect(getToggleEnableParameter(parameter.name)).toBeChecked();

                        toggleParameter(parameter.name);

                        expect(getToggleEnableParameter(parameter.name)).not.toBeChecked();

                        await userEvent.click(screen.getByRole('button', { name: `Reset ${parameter.name}` }));
                        expect(getToggleEnableParameter(parameter.name)).toBeChecked();
                    } else {
                        expect(getParameter(parameter.name)).toHaveValue(parameter.value.toString());

                        await userEvent.click(
                            screen.getByRole('button', { name: `Increase Change ${parameter.name}` })
                        );

                        expect(getParameter(parameter.name)).toHaveValue((Number(parameter.value) + 0.1).toString());

                        await userEvent.click(screen.getByRole('button', { name: `Reset ${parameter.name}` }));

                        expect(getParameter(parameter.name)).toHaveValue(parameter.default_value.toString());
                    }
                }
            }
        }
    });

    it('hides all dependent parameters when "DEIM framework" parameter is enabled', () => {
        render(<App dataAugmentationParameters={dataAugmentationParameters} />);

        expect(getToggleEnableParameter('DEIM framework')).not.toBeChecked();
        expect(screen.getByTestId('center_crop')).toBeVisible();
        expect(screen.getByTestId('random_affine')).toBeVisible();

        toggleParameter('DEIM framework');

        expect(getToggleEnableParameter('DEIM framework')).toBeChecked();
        expect(screen.queryByTestId('center_crop')).not.toBeInTheDocument();
        expect(screen.queryByTestId('random_affine')).not.toBeInTheDocument();
    });
});
