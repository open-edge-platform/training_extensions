// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { Flex } from '@geti/ui';
import { partition } from 'lodash-es';

import { ConfigurableParameter, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { isParameterGroup } from '../../../../model-listing/model-training-parameters/utils';
import { Parameter, Parameters, ParametersGroup } from '../../components/parameters.component';
import { deepReplaceParameter, replaceByKey } from '../../utils';
import { InputSizeParameters } from './input-size-parameters.component';
import {
    groupDependentParameters,
    isInputSizeParameter,
    LearningConfigurationGroup,
    LearningConfigurationParameters,
} from './utils';

const changeInputSizeParameters = (
    trainingConfiguration: TrainingConfiguration,
    newParameters: ConfigurableParameter[]
): TrainingConfiguration => {
    const parameters: TrainingConfiguration['parameters'] = replaceByKey(
        trainingConfiguration.parameters,
        'training',
        (parameterGroup) => {
            return {
                ...parameterGroup,
                parameters: parameterGroup.parameters.map((parameter) => {
                    const newParameter = newParameters.find(({ key }) => key === parameter.key);

                    return newParameter ?? parameter;
                }),
            };
        }
    );

    return { parameters };
};

type LearningParametersListProps = {
    learningParameters: LearningConfigurationParameters;
    onParameterChange: (parameter: ConfigurableParameter, groupKey: string) => void;
    isReadOnly: boolean;
};

const LearningParametersList = ({ learningParameters, onParameterChange, isReadOnly }: LearningParametersListProps) => {
    return <Parameters parameters={learningParameters} onChange={onParameterChange} isReadOnly={isReadOnly} />;

    /*  return (
        <Flex direction={'column'} gap={'size-300'}>
            {learningParameters.map((learningParameter) => {
                if (isParameterGroup(learningParameter)) {
                    const groupedParameters = groupDependentParameters(learningParameter.parameters);

                    return (
                        <ParametersGroup
                            key={learningParameter.key}
                            parametersGroup={learningParameter}
                            onChange={onParameterChange}
                        />
                    );
                }

                return (
                    <Parameters.Container key={learningParameter.key} isReadOnly={isReadOnly}>
                        <Parameter
                            header={learningParameter.name}
                            description={learningParameter.description}
                            parameter={learningParameter}
                            onChange={(parameter) => onParameterChange(parameter, learningParameter.key)}
                            isReadOnly={isReadOnly}
                        />
                    </Parameters.Container>
                );
            })}
        </Flex>
    );*/
};

type LearningParametersListContainerProps = {
    learningParameters: LearningConfigurationGroup;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
    isReadOnly?: boolean;
};

export const LearningParametersListContainer = ({
    isReadOnly = false,
    learningParameters,
    onTrainingConfigurationChange,
}: LearningParametersListContainerProps) => {
    const [inputSizeParameters, restParameters] = partition(learningParameters.parameters, isInputSizeParameter);

    const handleInputSizeParametersChange = (newParameters: ConfigurableParameter[]) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            return changeInputSizeParameters(config, newParameters);
        });
    };

    const handleParameterChange = (parameter: ConfigurableParameter, groupKey: string) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            const parameters: TrainingConfiguration['parameters'] = replaceByKey(
                config.parameters,
                'training',
                (group) => ({
                    ...group,
                    parameters: deepReplaceParameter(group.parameters, parameter, groupKey),
                })
            );

            return { parameters };
        });
    };

    return (
        <Flex direction={'column'} gap={'size-300'}>
            <InputSizeParameters
                inputSizeParameters={inputSizeParameters}
                onInputSizeParameterChange={handleInputSizeParametersChange}
                isReadOnly={isReadOnly}
            />
            <LearningParametersList
                learningParameters={restParameters}
                onParameterChange={handleParameterChange}
                isReadOnly={isReadOnly}
            />
        </Flex>
    );
};
