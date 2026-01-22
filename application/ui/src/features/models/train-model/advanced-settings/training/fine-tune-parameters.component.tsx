// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Checkbox, Flex, Radio, RadioGroup } from '@geti/ui';

import { Accordion } from '../ui/accordion/accordion.component';
import { Tooltip } from '../ui/tooltip.component';

import styles from './fine-tune-parameters.module.scss';

interface FineTuneParametersProps {
    trainFromScratch: boolean;
    onTrainFromScratchChange: (trainFromScratch: boolean) => void;

    onReshufflingSubsetsEnabledChange: (isChecked: boolean) => void;
    isReshufflingSubsetsEnabled: boolean;
}

enum TRAINING_WEIGHTS {
    PRE_TRAINED_WEIGHTS = 'Pre-trained weights',
    PREVIOUS_TRAINING_WEIGHTS = 'Previous training weights',
}

export const FineTuneParameters = ({
    trainFromScratch,
    onTrainFromScratchChange,
    isReshufflingSubsetsEnabled,
    onReshufflingSubsetsEnabledChange,
}: FineTuneParametersProps) => {
    const trainingWeight = trainFromScratch
        ? TRAINING_WEIGHTS.PRE_TRAINED_WEIGHTS
        : TRAINING_WEIGHTS.PREVIOUS_TRAINING_WEIGHTS;

    const handleTrainingWeightsChange = (value: string): void => {
        if (value === TRAINING_WEIGHTS.PRE_TRAINED_WEIGHTS) {
            onTrainFromScratchChange(true);
        } else {
            onTrainFromScratchChange(false);
        }
    };

    return (
        <Accordion>
            <Accordion.Title>
                Fine-tune parameters{' '}
                <Accordion.Tag ariaLabel={'Fine-tune parameters tag'}>{trainingWeight}</Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>
                    Fine-tuning is the process of adapting a pre-trained model as the starting point for learning new
                    tasks.
                </Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <RadioGroup label={'Training weights'} value={trainingWeight} onChange={handleTrainingWeightsChange}>
                    <Radio value={TRAINING_WEIGHTS.PREVIOUS_TRAINING_WEIGHTS}>
                        Previous training weights - fine-tune the previous version of your model
                    </Radio>
                    <Flex alignItems={'center'}>
                        <Radio value={TRAINING_WEIGHTS.PRE_TRAINED_WEIGHTS} marginEnd={'size-65'}>
                            Pre-trained weights - fine-tune the original model
                        </Radio>
                        <Tooltip>
                            <span>
                                The original model is a base version with pre-trained weights, already trained on a
                                large, general-purpose dataset. It provides a strong foundation for adapting to new,
                                specific tasks.
                            </span>
                        </Tooltip>
                    </Flex>
                </RadioGroup>

                <Flex gap={'size-100'} alignItems={'center'} marginTop={'size-100'}>
                    <Checkbox
                        isEmphasized
                        isSelected={isReshufflingSubsetsEnabled}
                        onChange={onReshufflingSubsetsEnabledChange}
                        UNSAFE_className={styles.trainModelCheckbox}
                        isDisabled={trainingWeight === TRAINING_WEIGHTS.PREVIOUS_TRAINING_WEIGHTS}
                    >
                        Reshuffle subsets
                    </Checkbox>

                    <Tooltip>
                        <span>
                            Reassign all dataset items to train, validation, and test subsets from scratch. Previous
                            splits will not be retained. This option is accessible for Pre-trained weights.
                        </span>
                    </Tooltip>
                </Flex>
            </Accordion.Content>
        </Accordion>
    );
};
