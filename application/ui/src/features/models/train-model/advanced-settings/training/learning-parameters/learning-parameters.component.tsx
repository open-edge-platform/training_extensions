// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import { isEqual } from 'lodash-es';

import { TrainingConfiguration } from '../../../../../../constants/shared-types';
import { Accordion } from '../../components/accordion/accordion.component';
import { LearningParametersList } from './learning-parameters-list.component';
import { LearningConfigurationParameters } from './utils';

type LearningParametersProps = {
    learningParameters: LearningConfigurationParameters;
    defaultLearningParameters?: LearningConfigurationParameters;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

export const LearningParameters = ({
    learningParameters,
    defaultLearningParameters,
    onTrainingConfigurationChange,
}: LearningParametersProps) => {
    const tag = isEqual(learningParameters, defaultLearningParameters) ? 'Default' : 'Modified';

    return (
        <Accordion>
            <Accordion.Title>
                Learning parameters
                <Accordion.Tag ariaLabel={'Learning parameters tag'}>{tag}</Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>{learningParameters.description}</Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <LearningParametersList
                    learningParameters={learningParameters}
                    onTrainingConfigurationChange={onTrainingConfigurationChange}
                />
            </Accordion.Content>
        </Accordion>
    );
};
