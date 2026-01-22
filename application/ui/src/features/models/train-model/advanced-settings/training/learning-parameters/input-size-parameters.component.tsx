// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Item, Picker, Text } from '@geti/ui';

import { EnumConfigurationParameter, TrainingConfiguration } from '../../../../configuration.interface';
import { Parameter, Parameters } from '../../ui/parameters.component';

interface InputSizeParameterProps {
    isReadOnly?: boolean;
    parameter: EnumConfigurationParameter;
    onChange: (parameter: EnumConfigurationParameter) => void;
}

const InputSizeParameter = ({ parameter, onChange, isReadOnly }: InputSizeParameterProps) => {
    if (isReadOnly) {
        return <span aria-label={parameter.name}>{parameter.value}</span>;
    }

    const items = parameter.allowed_values.map((value) => ({ value }));

    const handleSelectionChange = (value: string) => {
        onChange({
            ...parameter,
            value: Number(value),
        });
    };

    return (
        <Picker
            items={items}
            selectedKey={parameter.value.toString()}
            onSelectionChange={(key) => handleSelectionChange(key as string)}
            aria-label={`Select ${parameter.name}`}
            width={'size-1250'}
        >
            {(item) => (
                <Item key={item.value}>
                    <Text>{item.value}</Text>
                </Item>
            )}
        </Picker>
    );
};

export const INPUT_SIZE_WIDTH_KEY = 'input_size_width';
export const INPUT_SIZE_HEIGHT_KEY = 'input_size_height';
export const NUMBER_OF_INPUT_SIZE_PARAMETERS = 2;

type InputSizeParametersProps = {
    inputSizeParameters: EnumConfigurationParameter[];
    onUpdateTrainingConfiguration: (
        updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
    ) => void;
    isReadOnly?: boolean;
};

export const InputSizeParameters = ({
    isReadOnly,
    inputSizeParameters,
    onUpdateTrainingConfiguration,
}: InputSizeParametersProps) => {
    const inputSizeWidthParameter = inputSizeParameters.find(
        ({ key }) => key === INPUT_SIZE_WIDTH_KEY
    ) as EnumConfigurationParameter;
    const inputSizeHeightParameter = inputSizeParameters.find(
        ({ key }) => key === INPUT_SIZE_HEIGHT_KEY
    ) as EnumConfigurationParameter;

    const description = `${inputSizeWidthParameter.description}\n${inputSizeHeightParameter.description}`;

    const handleInputSizeChange = (parameter: EnumConfigurationParameter) => {
        onUpdateTrainingConfiguration((config) => {
            if (config === undefined) return undefined;

            const newConfig = structuredClone(config);
            newConfig.training = config.training.map((trainingParameter) => {
                if (trainingParameter.key === parameter.key) {
                    return parameter;
                }
                return trainingParameter;
            });

            return newConfig;
        });
    };

    const handleReset = () => {
        onUpdateTrainingConfiguration((config) => {
            if (config === undefined) return undefined;

            const newConfig = structuredClone(config);
            newConfig.training = config.training.map((trainingParameter) => {
                if (trainingParameter.key === inputSizeWidthParameter.key) {
                    return { ...inputSizeWidthParameter, value: inputSizeWidthParameter.default_value };
                }
                if (trainingParameter.key === inputSizeHeightParameter.key) {
                    return { ...inputSizeHeightParameter, value: inputSizeHeightParameter.default_value };
                }
                return trainingParameter;
            });

            return newConfig;
        });
    };

    return (
        <Parameters.Container>
            <Parameter.Layout
                header={'Input size'}
                description={description}
                onReset={isReadOnly ? undefined : handleReset}
            >
                <Flex alignItems={'center'} gap={'size-50'}>
                    <InputSizeParameter
                        isReadOnly={isReadOnly}
                        parameter={inputSizeWidthParameter}
                        onChange={handleInputSizeChange}
                    />
                    x
                    <InputSizeParameter
                        isReadOnly={isReadOnly}
                        parameter={inputSizeHeightParameter}
                        onChange={handleInputSizeChange}
                    />
                </Flex>
            </Parameter.Layout>
        </Parameters.Container>
    );
};
