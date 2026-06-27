// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, useMemo } from 'react';

import { Flex } from '@geti/ui';

import { ConfigurableParameter, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { Parameters } from '../../components/parameters.component';
import { deepReplaceParameters, filterDependentParameters } from '../../utils';
import { EvaluationConfigurationGroup } from './utils';

const changeEvaluationParameters = (
    trainingConfiguration: TrainingConfiguration,
    newParameters: ConfigurableParameter[],
    groupKeys: string[]
): TrainingConfiguration => {
    const parameters: TrainingConfiguration['parameters'] = deepReplaceParameters(
        trainingConfiguration.parameters,
        newParameters,
        ['evaluation', ...groupKeys]
    );

    return { parameters };
};

type EvaluationParametersListContainerProps = {
    evaluationParameters: EvaluationConfigurationGroup;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
    isReadOnly?: boolean;
};

export const EvaluationParametersListContainer = ({
    isReadOnly = false,
    evaluationParameters,
    onTrainingConfigurationChange,
}: EvaluationParametersListContainerProps) => {
    const evaluationParametersBasedOnDependency = useMemo(
        () => filterDependentParameters(evaluationParameters.parameters),
        [evaluationParameters.parameters]
    );

    const handleEvaluationParametersChange = (updatedParameters: ConfigurableParameter[], groupKeys: string[]) => {
        onTrainingConfigurationChange((config) => {
            if (config === undefined) return;

            return changeEvaluationParameters(config, updatedParameters, groupKeys);
        });
    };

    const handleEvaluationParameterChange = (newParameter: ConfigurableParameter, groupKeys: string[] = []) => {
        handleEvaluationParametersChange([newParameter], groupKeys);
    };

    return (
        <Flex direction={'column'} gap={'size-300'}>
            <Parameters
                parameters={evaluationParametersBasedOnDependency}
                onChange={handleEvaluationParameterChange}
                isReadOnly={isReadOnly}
            />
        </Flex>
    );
};
