// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { useState } from 'react';

import { fireEvent, screen } from '@testing-library/react';

import {
    ConfigurationParameter,
    NumberParameter,
    TrainingConfiguration,
} from '../../../../../../../../core/configurable-parameters/services/configuration.interface';
import { isBoolParameter, isConfigurationParameter } from '../../../../../../../../core/configurable-parameters/utils';
import {
    getMockedConfigurationParameter,
    getMockedTrainingConfiguration,
} from '../../../../../../../../test-utils/mocked-items-factory/mocked-configuration-parameters';
import { providersRender as render } from '../../../../../../../../test-utils/required-providers-render';
import { LEARNING_RATE_STEP } from './learning-parameters-list.component';
import { LearningParameters } from './learning-parameters.component';

type LearningParametersType = TrainingConfiguration['training'];

const getParameter = (name: string) => {
    return screen.getByRole('textbox', { name: `Change ${name}` });
};

const getToggleEnableParameter = (name: string) => {
    return screen.getByRole('switch', { name: `Toggle ${name}` });
};

const toggleParameter = (name: string) => {
    fireEvent.click(getToggleEnableParameter(name));
};

const resetParameter = (name: string) => {
    fireEvent.click(screen.getByRole('button', { name: `Reset ${name}` }));
};

const getInputSizeParameter = (name: string) => {
    return screen.getByRole('button', { name: new RegExp(`Select ${name}`) });
};

const expectParameterToUpdateProperly = (parameter: ConfigurationParameter) => {
    if (isBoolParameter(parameter)) {
        expect(getToggleEnableParameter(parameter.name)).toBeChecked();

        toggleParameter(parameter.name);

        expect(getToggleEnableParameter(parameter.name)).not.toBeChecked();

        resetParameter(parameter.name);
        expect(getToggleEnableParameter(parameter.name)).toBeChecked();
    } else {
        const isLearningRate = parameter.key === 'learning_rate';
        const step = parameter.type === 'float' ? (isLearningRate ? LEARNING_RATE_STEP : 0.001) : 1;

        expect(getParameter(parameter.name)).toHaveValue(parameter.value.toString());

        fireEvent.click(screen.getByRole('button', { name: `Increase Change ${parameter.name}` }));

        expect(getParameter(parameter.name)).toHaveValue((Number(parameter.value) + step).toString());

        resetParameter(parameter.name);
        expect(getParameter(parameter.name)).toHaveValue(parameter.defaultValue.toString());
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
            defaultValue: 500,
            maxValue: null,
            minValue: 0,
        }),
        getMockedConfigurationParameter({
            key: 'learning_rate',
            type: 'float',
            name: 'Learning rate',
            value: 0.004,
            description: 'Base learning rate for the optimizer',
            defaultValue: 0.001,
            maxValue: 1,
            minValue: 0,
        }),
        {
            early_stopping: [
                getMockedConfigurationParameter({
                    key: 'enable',
                    type: 'bool',
                    name: 'Enable early stopping',
                    value: true,
                    description: 'Whether to stop training early when performance stops improving',
                    defaultValue: true,
                }),
                getMockedConfigurationParameter({
                    key: 'patience',
                    type: 'int',
                    name: 'Patience',
                    value: 10,
                    description: 'Number of epochs with no improvement after which training will be stopped',
                    defaultValue: 1,
                    maxValue: null,
                    minValue: 0,
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

    it('updates tag to "Modified" when at least one parameter is changed, otherwise is "Default"', () => {
        render(<App learningParameters={learningParameters} />);

        const parameter = learningParameters[0] as NumberParameter;

        expect(screen.getByLabelText('Learning parameters tag')).toHaveTextContent('Default');

        expect(getParameter(parameter.name)).toHaveValue(parameter.value.toString());
        fireEvent.click(screen.getByRole('button', { name: `Increase Change ${parameter.name}` }));

        expect(getParameter(parameter.name)).toHaveValue((parameter.value + 1).toString());
        expect(screen.getByLabelText('Learning parameters tag')).toHaveTextContent('Modified');
    });

    it('updates parameters and resets them to default properly', () => {
        render(<App learningParameters={learningParameters} />);

        Object.values(learningParameters).forEach((parameter) => {
            if (isConfigurationParameter(parameter)) {
                expectParameterToUpdateProperly(parameter);
            } else {
                Object.values(parameter).forEach((groupParameter) => {
                    groupParameter.forEach((param) => {
                        expectParameterToUpdateProperly(param);
                    });
                });
            }
        });
    });

    it('updates input size parameters', () => {
        const inputSizeWidthParameter = getMockedConfigurationParameter({
            type: 'enum',
            allowedValues: [256, 512, 1024],
            value: 512,
            key: 'input_size_width',
            name: 'Input size width',
        });

        const inputSizeHeightParameter = getMockedConfigurationParameter({
            type: 'enum',
            allowedValues: [256, 512, 1024],
            value: 512,
            key: 'input_size_height',
            name: 'Input size height',
        });

        const parameters = [inputSizeWidthParameter, inputSizeHeightParameter];

        render(<App learningParameters={parameters} />);

        parameters.forEach((parameter) => {
            const parameterSelector = getInputSizeParameter(parameter.name);

            expect(parameterSelector).toHaveTextContent(parameter.value.toString());

            fireEvent.click(parameterSelector);

            parameter.allowedValues.forEach((value) => {
                expect(screen.getByRole('option', { name: value.toString() })).toBeInTheDocument();
            });

            fireEvent.keyDown(screen.getByRole('listbox', { name: `Select ${parameter.name}` }), { key: 'Escape' });
        });

        fireEvent.click(getInputSizeParameter(inputSizeWidthParameter.name));
        fireEvent.click(screen.getByRole('option', { name: '1024' }));
        expect(getInputSizeParameter(inputSizeWidthParameter.name)).toHaveTextContent('1024');

        fireEvent.click(getInputSizeParameter(inputSizeHeightParameter.name));
        fireEvent.click(screen.getByRole('option', { name: '256' }));
        expect(getInputSizeParameter(inputSizeHeightParameter.name)).toHaveTextContent('256');
    });

    it('does not display input size parameters if there is only one of them', () => {
        const inputSizeWidthParameter = getMockedConfigurationParameter({
            type: 'enum',
            allowedValues: [256, 512, 1024],
            value: 512,
            key: 'input_size_width',
            name: 'Input size width',
        });

        const parameters = [inputSizeWidthParameter];

        render(<App learningParameters={parameters} />);

        expect(screen.queryByRole('button', { name: new RegExp(`Select Input size width`) })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: new RegExp(`Select Input size height`) })).not.toBeInTheDocument();
    });

    it('resets both input size parameters to default values', () => {
        const inputSizeWidthParameter = getMockedConfigurationParameter({
            type: 'enum',
            allowedValues: [256, 512, 1024],
            value: 512,
            defaultValue: 256,
            key: 'input_size_width',
            name: 'Input size width',
        });

        const inputSizeHeightParameter = getMockedConfigurationParameter({
            type: 'enum',
            allowedValues: [256, 512, 1024],
            value: 512,
            defaultValue: 256,
            key: 'input_size_height',
            name: 'Input size height',
        });

        const parameters = [inputSizeWidthParameter, inputSizeHeightParameter];

        render(<App learningParameters={parameters} />);

        fireEvent.click(getInputSizeParameter(inputSizeWidthParameter.name));
        fireEvent.click(screen.getByRole('option', { name: '1024' }));

        expect(getInputSizeParameter(inputSizeWidthParameter.name)).toHaveTextContent('1024');

        fireEvent.click(getInputSizeParameter(inputSizeHeightParameter.name));
        fireEvent.click(screen.getByRole('option', { name: '1024' }));

        expect(getInputSizeParameter(inputSizeHeightParameter.name)).toHaveTextContent('1024');

        fireEvent.click(screen.getByRole('button', { name: 'Reset Input size' }));

        expect(getInputSizeParameter(inputSizeWidthParameter.name)).toHaveTextContent(
            inputSizeWidthParameter.defaultValue.toString()
        );
        expect(getInputSizeParameter(inputSizeHeightParameter.name)).toHaveTextContent(
            inputSizeHeightParameter.defaultValue.toString()
        );
    });
});
