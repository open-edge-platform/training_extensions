// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedConfigurationParameter, getMockedTrainingConfiguration } from 'mocks/mock-training-configuration';
import { render } from 'test-utils/render';

import { ConfigurationParameter, NumberParameter, TrainingConfiguration } from '../../../../configuration.interface';
import { isBoolParameter, isConfigurationParameter } from '../../utils';
import { LEARNING_RATE_STEP } from './learning-parameters-list.component';
import { LearningParameters } from './learning-parameters.component';

type LearningParametersType = TrainingConfiguration['training'];

const getParameter = (name: string) => {
    return screen.getByRole('textbox', { name: `Change ${name}` });
};

const getToggleEnableParameter = (name: string) => {
    return screen.getByRole('switch', { name: `Toggle ${name}` });
};

const toggleParameter = async (name: string) => {
    await userEvent.click(getToggleEnableParameter(name));
};

const resetParameter = async (name: string) => {
    await userEvent.click(screen.getByRole('button', { name: `Reset ${name}` }));
};

const getInputSizeParameter = (name: string) => {
    return screen.getByRole('button', { name: new RegExp(`Select ${name}`) });
};

const expectParameterToUpdateProperly = async (parameter: ConfigurationParameter) => {
    if (isBoolParameter(parameter)) {
        expect(getToggleEnableParameter(parameter.name)).toBeChecked();

        await toggleParameter(parameter.name);

        expect(getToggleEnableParameter(parameter.name)).not.toBeChecked();

        await resetParameter(parameter.name);
        expect(getToggleEnableParameter(parameter.name)).toBeChecked();
    } else {
        const isLearningRate = parameter.key === 'learning_rate';
        const step = parameter.type === 'float' ? (isLearningRate ? LEARNING_RATE_STEP : 0.001) : 1;

        expect(getParameter(parameter.name)).toHaveValue(parameter.value.toString());

        await userEvent.click(screen.getByRole('button', { name: `Increase Change ${parameter.name}` }));

        expect(getParameter(parameter.name)).toHaveValue((Number(parameter.value) + step).toString());

        await resetParameter(parameter.name);
        expect(getParameter(parameter.name)).toHaveValue(parameter.default_value.toString());
    }
};

describe('LearningParameters', () => {
    const learningParameters: LearningParametersType = [
        getMockedConfigurationParameter({
            key: 'max_epochs',
            type: 'int',
            name: 'Maximum epochs',
            value: 200,
            description: 'Maximum number of training epochs to run',
            default_value: 500,
            max_value: null,
            min_value: 0,
        }),
        getMockedConfigurationParameter({
            key: 'learning_rate',
            type: 'float',
            name: 'Learning rate',
            value: 0.004,
            description: 'Base learning rate for the optimizer',
            default_value: 0.001,
            max_value: 1,
            min_value: 0,
        }),
        {
            early_stopping: [
                getMockedConfigurationParameter({
                    key: 'enable',
                    type: 'bool',
                    name: 'Enable early stopping',
                    value: true,
                    description: 'Whether to stop training early when performance stops improving',
                    default_value: true,
                }),
                getMockedConfigurationParameter({
                    key: 'patience',
                    type: 'int',
                    name: 'Patience',
                    value: 10,
                    description: 'Number of epochs with no improvement after which training will be stopped',
                    default_value: 1,
                    max_value: null,
                    min_value: 0,
                }),
            ],
        },
    ];

    const App = (props: { learningParameters: LearningParametersType }) => {
        const [trainingConfiguration, setTrainingConfiguration] = useState<TrainingConfiguration | undefined>(() =>
            getMockedTrainingConfiguration({
                training: props.learningParameters,
            })
        );

        const handleUpdateTrainingConfiguration = (
            updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
        ) => {
            setTrainingConfiguration(updateFunction);
        };

        return (
            <LearningParameters
                parameters={trainingConfiguration?.training ?? props.learningParameters}
                defaultParameters={props.learningParameters}
                onUpdateTrainingConfiguration={handleUpdateTrainingConfiguration}
            />
        );
    };

    it('updates tag to "Modified" when at least one parameter is changed, otherwise is "Default"', async () => {
        render(<App learningParameters={learningParameters} />);

        const parameter = learningParameters[0] as NumberParameter;

        expect(screen.getByLabelText('Learning parameters tag')).toHaveTextContent('Default');

        expect(getParameter(parameter.name)).toHaveValue(parameter.value.toString());
        await userEvent.click(screen.getByRole('button', { name: `Increase Change ${parameter.name}` }));

        expect(getParameter(parameter.name)).toHaveValue((parameter.value + 1).toString());
        expect(screen.getByLabelText('Learning parameters tag')).toHaveTextContent('Modified');
    });

    it('updates parameters and resets them to default properly', async () => {
        render(<App learningParameters={learningParameters} />);

        for (const parameter of Object.values(learningParameters)) {
            if (isConfigurationParameter(parameter)) {
                await expectParameterToUpdateProperly(parameter);
            } else {
                for (const groupParameter of Object.values(parameter)) {
                    for (const param of groupParameter) {
                        await expectParameterToUpdateProperly(param);
                    }
                }
            }
        }
    });

    it('updates input size parameters', async () => {
        const inputSizeWidthParameter = getMockedConfigurationParameter({
            type: 'enum',
            allowed_values: [256, 512, 1024],
            value: 512,
            key: 'input_size_width',
            name: 'Input size width',
        });

        const inputSizeHeightParameter = getMockedConfigurationParameter({
            type: 'enum',
            allowed_values: [256, 512, 1024],
            value: 512,
            key: 'input_size_height',
            name: 'Input size height',
        });

        const parameters = [inputSizeWidthParameter, inputSizeHeightParameter];

        render(<App learningParameters={parameters} />);

        for (const parameter of parameters) {
            const parameterSelector = getInputSizeParameter(parameter.name);

            expect(parameterSelector).toHaveTextContent(parameter.value.toString());

            await userEvent.click(parameterSelector);

            parameter.allowed_values.forEach((value) => {
                expect(screen.getByRole('option', { name: value.toString() })).toBeInTheDocument();
            });

            fireEvent.keyDown(screen.getByRole('listbox', { name: `Select ${parameter.name}` }), {
                key: 'Escape',
            });
        }

        await userEvent.click(getInputSizeParameter(inputSizeWidthParameter.name));
        await userEvent.click(screen.getByRole('option', { name: '1024' }));
        expect(getInputSizeParameter(inputSizeWidthParameter.name)).toHaveTextContent('1024');

        await userEvent.click(getInputSizeParameter(inputSizeHeightParameter.name));
        await userEvent.click(screen.getByRole('option', { name: '256' }));
        expect(getInputSizeParameter(inputSizeHeightParameter.name)).toHaveTextContent('256');
    });

    it('does not display input size parameters if there is only one of them', () => {
        const inputSizeWidthParameter = getMockedConfigurationParameter({
            type: 'enum',
            allowed_values: [256, 512, 1024],
            value: 512,
            key: 'input_size_width',
            name: 'Input size width',
        });

        const parameters = [inputSizeWidthParameter];

        render(<App learningParameters={parameters} />);

        expect(screen.queryByRole('button', { name: new RegExp(`Select Input size width`) })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: new RegExp(`Select Input size height`) })).not.toBeInTheDocument();
    });

    it('resets both input size parameters to default values', async () => {
        const inputSizeWidthParameter = getMockedConfigurationParameter({
            type: 'enum',
            allowed_values: [256, 512, 1024],
            value: 512,
            default_value: 256,
            key: 'input_size_width',
            name: 'Input size width',
        });

        const inputSizeHeightParameter = getMockedConfigurationParameter({
            type: 'enum',
            allowed_values: [256, 512, 1024],
            value: 512,
            default_value: 256,
            key: 'input_size_height',
            name: 'Input size height',
        });

        const parameters = [inputSizeWidthParameter, inputSizeHeightParameter];

        render(<App learningParameters={parameters} />);

        await userEvent.click(getInputSizeParameter(inputSizeWidthParameter.name));
        await userEvent.click(screen.getByRole('option', { name: '1024' }));

        expect(getInputSizeParameter(inputSizeWidthParameter.name)).toHaveTextContent('1024');

        await userEvent.click(getInputSizeParameter(inputSizeHeightParameter.name));
        await userEvent.click(screen.getByRole('option', { name: '1024' }));

        expect(getInputSizeParameter(inputSizeHeightParameter.name)).toHaveTextContent('1024');

        await userEvent.click(screen.getByRole('button', { name: 'Reset Input size' }));

        expect(getInputSizeParameter(inputSizeWidthParameter.name)).toHaveTextContent(
            inputSizeWidthParameter.default_value.toString()
        );
        expect(getInputSizeParameter(inputSizeHeightParameter.name)).toHaveTextContent(
            inputSizeHeightParameter.default_value.toString()
        );
    });
});
