// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, useMemo } from 'react';

import { Flex } from '@geti-ui/ui';
import { partition } from 'lodash-es';

import { ConfigurableParameter, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { Parameters } from '../../components/parameters.component';
import { deepReplaceParameters, filterDependentParameters } from '../../utils';
import { InputSizeParameters } from './input-size-parameters.component';
import { isInputSizeParameter, LearningConfigurationGroup } from './utils';

const changeLearningParameters = (
    trainingConfiguration: TrainingConfiguration,
    newParameters: ConfigurableParameter[],
    groupKeys: string[]
): TrainingConfiguration => {
    const parameters: TrainingConfiguration['parameters'] = deepReplaceParameters(
        trainingConfiguration.parameters,
        newParameters,
        ['training', ...groupKeys]
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
        () => filterDependentParameters(restParameters),
        [restParameters]
    );

    const handleLearningParametersChange = (updatedParameters: ConfigurableParameter[], groupKeys: string[]) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            return changeLearningParameters(config, updatedParameters, groupKeys);
        });
    };

    const handleLearningParameterChange = (newParameter: ConfigurableParameter, groupKeys: string[] = []) => {
        handleLearningParametersChange([newParameter], groupKeys);
    };

    const handleInputSizeParametersChange = (updatedParameters: ConfigurableParameter[]) => {
        handleLearningParametersChange(updatedParameters, []);
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
                onChange={handleLearningParameterChange}
                isReadOnly={isReadOnly}
            />
        </Flex>
    );
};
