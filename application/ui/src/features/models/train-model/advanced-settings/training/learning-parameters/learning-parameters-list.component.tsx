// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { Flex } from '@geti/ui';
import { partition } from 'lodash-es';

import { ConfigurableParameter, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { replaceByKey } from '../../utils';
import { InputSizeParameters } from './input-size-parameters.component';
import { isInputSizeParameter, LearningConfigurationParameters } from './utils';

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
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
    isReadOnly?: boolean;
};

export const LearningParametersList = ({
    isReadOnly = false,
    learningParameters,
    onTrainingConfigurationChange,
}: LearningParametersListProps) => {
    const [inputSizeParameters, restParameters] = partition(learningParameters.parameters, isInputSizeParameter);

    const handleInputSizeParametersChange = (newParameters: ConfigurableParameter[]) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            return changeInputSizeParameters(config, newParameters);
        });
    };

    return (
        <Flex direction={'column'} gap={'size-300'}>
            <InputSizeParameters
                inputSizeParameters={inputSizeParameters}
                onInputSizeParameterChange={handleInputSizeParametersChange}
                isReadOnly={isReadOnly}
            />
        </Flex>
    );
};
