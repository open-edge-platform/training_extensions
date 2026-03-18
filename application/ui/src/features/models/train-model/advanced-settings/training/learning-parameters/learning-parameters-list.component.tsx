// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, useMemo } from 'react';

import { Flex } from '@geti/ui';
import { partition } from 'lodash-es';

import { ConfigurableParameter, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { Parameters } from '../../components/parameters.component';
import { deepReplaceParameter, replaceByKey } from '../../utils';
import { InputSizeParameters } from './input-size-parameters.component';
import { groupDependentParameters, isInputSizeParameter, LearningConfigurationGroup } from './utils';

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

    const learningParametersBasedOnDependency = useMemo(
        () => groupDependentParameters(restParameters),
        [restParameters]
    );

    const handleInputSizeParametersChange = (newParameters: ConfigurableParameter[]) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            return changeInputSizeParameters(config, newParameters);
        });
    };

    const handleParameterChange = (parameter: ConfigurableParameter, groupKeys?: string[]) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            const parameters: TrainingConfiguration['parameters'] = replaceByKey(
                config.parameters,
                'training',
                (group) => ({
                    ...group,
                    parameters: deepReplaceParameter(group.parameters, parameter, groupKeys),
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
            <Parameters
                parameters={learningParametersBasedOnDependency}
                onChange={handleParameterChange}
                isReadOnly={isReadOnly}
            />
        </Flex>
    );
};
