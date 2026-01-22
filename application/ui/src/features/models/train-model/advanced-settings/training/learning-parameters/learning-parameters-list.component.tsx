// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { noop, partition } from 'lodash-es';

import { ConfigurationParameter, NumberParameter, TrainingConfiguration } from '../../../../configuration.interface';
import { NumberParameterField } from '../../ui/number-parameter-field.component';
import { Parameter, Parameters } from '../../ui/parameters.component';
import { isConfigurationParameter, isEnumParameter } from '../../utils';
import {
    INPUT_SIZE_HEIGHT_KEY,
    INPUT_SIZE_WIDTH_KEY,
    InputSizeParameters,
    NUMBER_OF_INPUT_SIZE_PARAMETERS,
} from './input-size-parameters.component';

export type LearningParametersType = TrainingConfiguration['training'];

interface LearningParametersListProps {
    parameters: LearningParametersType;
    onUpdateTrainingConfiguration?: (
        updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
    ) => void;
    isReadOnly?: boolean;
}

interface SingleLearningParameterProps {
    parameter: ConfigurationParameter;
    onUpdateTrainingConfiguration: (
        updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
    ) => void;
    isReadOnly: boolean;
}

const isLearningRateParameter = (parameter: ConfigurationParameter): parameter is NumberParameter => {
    return parameter.type === 'float' && parameter.key === 'learning_rate';
};

export const LEARNING_RATE_STEP = 1e-6;

const SingleLearningParameter = ({
    parameter,
    onUpdateTrainingConfiguration,
    isReadOnly,
}: SingleLearningParameterProps) => {
    const handleChange = (inputParameter: ConfigurationParameter) => {
        onUpdateTrainingConfiguration((config) => {
            if (!config) return undefined;

            const newConfig = structuredClone(config);

            newConfig.training = config.training.map((trainingParameter) => {
                if (trainingParameter.key === inputParameter.key) {
                    return inputParameter;
                }

                return trainingParameter;
            });

            return newConfig;
        });
    };

    if (isReadOnly) {
        return <Parameters key={parameter.key} parameters={[parameter]} onChange={handleChange} isReadOnly />;
    }

    if (isLearningRateParameter(parameter)) {
        const handleLearningRateChange = (value: number) => {
            handleChange({
                ...parameter,
                value,
            });
        };

        const handleLearningRateReset = () => {
            handleChange({
                ...parameter,
                value: parameter.defaultValue,
            });
        };

        return (
            <Parameters.Container>
                <Parameter.Layout
                    description={parameter.description}
                    header={parameter.name}
                    onReset={handleLearningRateReset}
                >
                    <NumberParameterField
                        onChange={handleLearningRateChange}
                        isDisabled={isReadOnly}
                        value={parameter.value}
                        name={parameter.name}
                        type={parameter.type}
                        step={LEARNING_RATE_STEP}
                        maxValue={parameter.maxValue}
                        minValue={parameter.minValue}
                    />
                </Parameter.Layout>
            </Parameters.Container>
        );
    }

    return <Parameters key={parameter.key} parameters={[parameter]} onChange={handleChange} />;
};

type LearningParametersGroupProps = {
    groupKey: string;
    parameters: ConfigurationParameter[];
    onUpdateTrainingConfiguration: (
        updateFunction: (config: TrainingConfiguration | undefined) => TrainingConfiguration | undefined
    ) => void;
    isReadOnly: boolean;
};

const LearningParametersGroup = ({
    groupKey,
    parameters,
    isReadOnly,
    onUpdateTrainingConfiguration,
}: LearningParametersGroupProps) => {
    const handleChange = (inputParameter: ConfigurationParameter) => {
        onUpdateTrainingConfiguration((config) => {
            if (!config) return undefined;

            const newConfig = structuredClone(config);

            newConfig.training = config.training.map((trainingParameter) => {
                if (isConfigurationParameter(trainingParameter)) {
                    return trainingParameter;
                }

                if (trainingParameter[groupKey] === undefined) {
                    return trainingParameter;
                }

                return {
                    ...trainingParameter,
                    [groupKey]: trainingParameter[groupKey].map((trainingParam) =>
                        trainingParam.key === inputParameter.key ? inputParameter : trainingParam
                    ),
                };
            });

            return newConfig;
        });
    };

    return <Parameters parameters={parameters} onChange={handleChange} isReadOnly={isReadOnly} />;
};

export const LearningParametersList = ({
    parameters,
    onUpdateTrainingConfiguration = noop,
    isReadOnly = false,
}: LearningParametersListProps) => {
    const [inputSizeParameters, restParameters] = partition(
        parameters,
        (parameter) => parameter.key === INPUT_SIZE_WIDTH_KEY || parameter.key === INPUT_SIZE_HEIGHT_KEY
    );

    return (
        <Flex direction={'column'} width={'100%'} gap={'size-300'}>
            {inputSizeParameters.length === NUMBER_OF_INPUT_SIZE_PARAMETERS &&
                inputSizeParameters.every(isEnumParameter) && (
                    <InputSizeParameters
                        isReadOnly={isReadOnly}
                        inputSizeParameters={inputSizeParameters}
                        onUpdateTrainingConfiguration={onUpdateTrainingConfiguration}
                    />
                )}
            {restParameters.map((parameter) => {
                if (isConfigurationParameter(parameter)) {
                    return (
                        <SingleLearningParameter
                            key={parameter.key}
                            parameter={parameter}
                            onUpdateTrainingConfiguration={onUpdateTrainingConfiguration}
                            isReadOnly={isReadOnly}
                        />
                    );
                }

                const objectParameters: [string, ConfigurationParameter[]][] = Object.entries(parameter);

                return objectParameters.map(([key, parametersLocal]) => {
                    return (
                        <LearningParametersGroup
                            key={key}
                            groupKey={key}
                            parameters={parametersLocal}
                            onUpdateTrainingConfiguration={onUpdateTrainingConfiguration}
                            isReadOnly={isReadOnly}
                        />
                    );
                });
            })}
        </Flex>
    );
};
