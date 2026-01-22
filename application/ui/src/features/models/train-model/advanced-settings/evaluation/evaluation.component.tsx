// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Grid, Item, minmax, Picker, Text, ToggleButtons, View } from '@geti/ui';
import { noop } from 'lodash-es';

import { ConfigurationParameter, NumberParameter } from '../../../configuration.interface';
import { Accordion } from '../ui/accordion/accordion.component';
import { NumberParameterField } from '../ui/number-parameter-field.component';
import { ResetButton } from '../ui/reset-button.component';
import { Tooltip } from '../ui/tooltip.component';

import styles from './evaluation.module.scss';

type EvaluationProps = {
    evaluationParameters: ConfigurationParameter[];
};

const ConfidenceThresholdTooltip = () => {
    return (
        <Tooltip>
            Set the minimum confidence level for predictions. This threshold determines the certainty required for a
            prediction to be considered valid.
            <br />
            <b>Manual</b>: Specify a fixed confidence level. Higher values ensure more reliable predictions but may
            exclude uncertain cases. Lower values allow more flexibility but may include less certain predictions.
            <br />
            <b>Result based</b>: Let the system determine the optimal threshold based on evaluation metrics. This
            approach adapts to the model&#39;s performance, ensuring a balance between precision and recall.
        </Tooltip>
    );
};

type ConfidenceThresholdProps = {
    confidenceThresholdMode: CONFIDENCE_THRESHOLD_MODE;
    onConfidenceThresholdModeChange: (mode: CONFIDENCE_THRESHOLD_MODE) => void;
    manualParameter: NumberParameter;
    onChange: () => void;
};

const ConfidenceThreshold = ({
    manualParameter,
    onChange,
    confidenceThresholdMode,
    onConfidenceThresholdModeChange,
}: ConfidenceThresholdProps) => {
    const isManualMode = confidenceThresholdMode === CONFIDENCE_THRESHOLD_MODE.MANUAL;

    return (
        <Grid columns={['size-3000', minmax('size-3400', '1fr'), 'size-400']} alignItems={'center'} gap={'size-250'}>
            <Text gridColumn={'1/2'}>
                Confidence threshold <ConfidenceThresholdTooltip />
            </Text>
            <View>
                <ToggleButtons
                    options={[CONFIDENCE_THRESHOLD_MODE.RESULT_BASED, CONFIDENCE_THRESHOLD_MODE.MANUAL]}
                    selectedOption={confidenceThresholdMode}
                    onOptionChange={onConfidenceThresholdModeChange}
                />
            </View>
            {isManualMode && (
                <>
                    <View gridColumn={'2/3'}>
                        <NumberParameterField
                            name={manualParameter.name}
                            value={manualParameter.value}
                            minValue={manualParameter.minValue}
                            maxValue={manualParameter.maxValue}
                            onChange={onChange}
                            type={manualParameter.type}
                        />
                    </View>
                    <ResetButton aria-label={`Reset ${manualParameter.name}`} onPress={noop} />
                </>
            )}
        </Grid>
    );
};

enum CONFIDENCE_THRESHOLD_MODE {
    RESULT_BASED = 'Result based',
    MANUAL = 'Manual',
}

const getConfidenceThresholdMode = (evaluationParameters: ConfigurationParameter[]): CONFIDENCE_THRESHOLD_MODE => {
    const resultBased = evaluationParameters.find(
        (parameter) => parameter.key === '"result_based_confidence_threshold"'
    );

    if (resultBased?.value) {
        return CONFIDENCE_THRESHOLD_MODE.RESULT_BASED;
    }

    return CONFIDENCE_THRESHOLD_MODE.MANUAL;
};

const MODEL_ACCEPTANCE_TUNE_PARAMETERS = [
    {
        name: 'Accuracy',
    },
    {
        name: 'F1 score',
    },
    {
        name: 'Precision over recall',
    },
    {
        name: 'Recall over precision',
    },
    {
        name: 'Intersection over union',
    },
] as const;

type TuneParameter = (typeof MODEL_ACCEPTANCE_TUNE_PARAMETERS)[number]['name'];

/*
 * This component is currently not used in the v1 of Training flow revamp. Will be used in the later version.
 */
export const Evaluation = ({ evaluationParameters }: EvaluationProps) => {
    const [tuneParameter, setTuneParameter] = useState<TuneParameter>(MODEL_ACCEPTANCE_TUNE_PARAMETERS[0].name);
    const [confidenceThresholdMode, setConfidenceThresholdMode] = useState(() =>
        getConfidenceThresholdMode(evaluationParameters)
    );
    const manualParameter = evaluationParameters.find(
        (parameter) => parameter.name === 'confidence_threshold'
    ) as NumberParameter;

    return (
        <Accordion isExpanded UNSAFE_className={styles.accordionChevronIcon}>
            <Accordion.Title>Evaluation</Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>
                    Choose the method for calculating confidence threshold for your model as wel as acceptance criteria.
                </Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <ConfidenceThreshold
                    onChange={noop}
                    manualParameter={manualParameter}
                    confidenceThresholdMode={confidenceThresholdMode}
                    onConfidenceThresholdModeChange={setConfidenceThresholdMode}
                />
                <Accordion.Divider marginY={'size-250'} />
                <Picker
                    label={'Tune model acceptance based on'}
                    items={MODEL_ACCEPTANCE_TUNE_PARAMETERS}
                    selectedKey={tuneParameter}
                    onSelectionChange={(key) => setTuneParameter(key as TuneParameter)}
                >
                    {(item) => <Item key={item.name}>{item.name}</Item>}
                </Picker>
            </Accordion.Content>
        </Accordion>
    );
};
