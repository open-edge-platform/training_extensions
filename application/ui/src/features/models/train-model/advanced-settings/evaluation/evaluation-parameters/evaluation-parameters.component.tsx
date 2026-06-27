// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { isEqual } from 'lodash-es';

import { TrainingConfiguration } from '../../../../../../constants/shared-types';
import { Accordion } from '../../components/accordion/accordion.component';
import { EvaluationParametersListContainer } from './evaluation-parameters-list.component';
import { EvaluationConfigurationGroup } from './utils';

type EvaluationParametersProps = {
    evaluationParameters: EvaluationConfigurationGroup;
    defaultEvaluationParameters?: EvaluationConfigurationGroup;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

export const EvaluationParameters = ({
    evaluationParameters,
    defaultEvaluationParameters,
    onTrainingConfigurationChange,
}: EvaluationParametersProps) => {
    const tag = isEqual(evaluationParameters, defaultEvaluationParameters) ? 'Default' : 'Modified';

    return (
        <Accordion>
            <Accordion.Title>
                Evaluation parameters
                <Accordion.Tag ariaLabel={'Evaluation parameters tag'}>{tag}</Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>{evaluationParameters.description}</Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <EvaluationParametersListContainer
                    evaluationParameters={evaluationParameters}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            </Accordion.Content>
        </Accordion>
    );
};
