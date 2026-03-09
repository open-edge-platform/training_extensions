// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, useState } from 'react';

import { Content, Flex, Grid, Heading, InlineAlert, minmax, Text, View } from '@geti/ui';
import { isEqual } from 'lodash-es';

import type { ConfigurableParameter, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { isParameterGroup } from '../../../../model-listing/model-training-parameters/utils';
import { Accordion } from '../../components/accordion/accordion.component';
import { ResetButton } from '../../components/reset-button.component';
import { SubsetsDistributionSlider } from './subsets-distribution-slider/subsets-distribution-slider.component';
import {
    areSubsetsSizesValid,
    getSubsets,
    getSubsetsSizes,
    MAX_RATIO_VALUE,
    SubsetSplitParameters,
    TEST_SUBSET_KEY,
    TRAINING_SUBSET_KEY,
    VALIDATION_SUBSET_KEY,
} from './utils';

import classes from './training-subsets.module.scss';

type SubsetDistributionStatsProps = {
    trainingSize: number;
    validationSize: number;
    testSize: number;
};

const Tile = ({ color }: { color: string }) => {
    return (
        <View height={'size-100'} width={'size-100'} borderRadius={'small'} UNSAFE_style={{ backgroundColor: color }} />
    );
};

const SubsetDistributionStat = ({ size, color, title }: { size: number; color: string; title: string }) => {
    return (
        <Flex alignItems={'center'} gap={'size-50'}>
            <Tile color={color} />
            <span aria-label={`${title} subset size`}>
                {title}: {size}
            </span>
        </Flex>
    );
};

const SubsetDistributionStats = ({ trainingSize, validationSize, testSize }: SubsetDistributionStatsProps) => {
    return (
        <View gridArea={'counts'} backgroundColor={'static-gray-800'} borderRadius={'small'} padding={'size-100'}>
            <Flex alignItems={'center'} justifyContent={'space-between'} UNSAFE_className={classes.statsText}>
                <Flex alignItems={'center'} gap={'size-200'}>
                    <SubsetDistributionStat title={'Training'} color={'var(--training-subset)'} size={trainingSize} />
                    <SubsetDistributionStat
                        title={'Validation'}
                        color={'var(--validation-subset)'}
                        size={validationSize}
                    />
                    <SubsetDistributionStat title={'Test'} color={'var(--test-subset)'} size={testSize} />
                </Flex>
                <Text>
                    <Text UNSAFE_className={classes.totalStats}>Total: </Text>
                    {trainingSize + validationSize + testSize} media items
                </Text>
            </Flex>
        </View>
    );
};

type SubsetsDistributionProps = {
    trainingSubsetSize: number;
    validationSubsetSize: number;
    testSubsetSize: number;
    subsetsDistribution: number[];
    onSubsetsDistributionChange: (values: number[]) => void;
    onSubsetsDistributionChangeEnd: (values: number[]) => void;
    onSubsetsDistributionReset: () => void;
};

const SubsetsDistribution = ({
    subsetsDistribution,
    trainingSubsetSize,
    testSubsetSize,
    validationSubsetSize,
    onSubsetsDistributionChange,
    onSubsetsDistributionChangeEnd,
    onSubsetsDistributionReset,
}: SubsetsDistributionProps) => {
    const handleSubsetDistributionChange = (values: number[] | number): void => {
        if (Array.isArray(values)) {
            onSubsetsDistributionChange(values);
        }
    };

    const handleSubsetDistributionChangeEnd = (values: number[] | number): void => {
        if (Array.isArray(values)) {
            onSubsetsDistributionChangeEnd(values);
        }
    };

    return (
        <View UNSAFE_className={classes.trainingSubsets}>
            <Grid
                areas={['label slider reset', '. counts .']}
                columns={['max-content', minmax('size-3400', '1fr'), 'max-content']}
                alignItems={'center'}
                columnGap={'size-250'}
            >
                <SubsetsDistributionSlider
                    aria-label={'Distribute samples'}
                    minValue={0}
                    maxValue={100}
                    step={1}
                    value={[subsetsDistribution[0], subsetsDistribution[1]]}
                    onChange={handleSubsetDistributionChange}
                    onChangeEnd={handleSubsetDistributionChangeEnd}
                    label={'Distribution'}
                />
                <ResetButton
                    gridArea={'reset'}
                    onPress={onSubsetsDistributionReset}
                    aria-label={'Reset training subsets'}
                />
                <SubsetDistributionStats
                    testSize={testSubsetSize}
                    trainingSize={trainingSubsetSize}
                    validationSize={validationSubsetSize}
                />
            </Grid>
        </View>
    );
};

type TrainingSubsetsProps = {
    defaultSubsetParameters: SubsetSplitParameters;
    hasSupportedModels: boolean;
    subsetsParameters: SubsetSplitParameters;
    onTrainingConfigurationChange: Dispatch<SetStateAction<TrainingConfiguration | undefined>>;
};

const TrainingSubsetsUnavailable = () => {
    return (
        <InlineAlert variant={'notice'}>
            <Heading>Invalid training subsets configuration</Heading>
            <Content>
                Training subsets do not contain enough media items to support a configurable split between training,
                validation, and testing subsets.
                <br />
                Please add more media items to ensure each subset contains at least one item.
            </Content>
        </InlineAlert>
    );
};

const TrainingSubsetsChangedDistributionWarning = () => {
    return (
        <InlineAlert variant={'notice'}>
            <Heading>Additional configuration change required to apply new training subsets distribution</Heading>
            <Content>
                To apply the updated distribution of training, validation, and testing subsets, please go to{' '}
                {'"Training"'} tab, choose {'"Pre-trained weights"'}, and enable {'"Reshuffle subsets"'}.
                <br />
                This will reset your data splits and begin a new training process, replacing the current model.
            </Content>
        </InlineAlert>
    );
};

const updateSubsetSplitValues = (
    config: TrainingConfiguration,
    getValueForKey: (parameter: ConfigurableParameter) => number
): TrainingConfiguration => ({
    parameters: config.parameters.map((parameterGroup) => {
        if (parameterGroup.key !== 'dataset_preparation' || !isParameterGroup(parameterGroup)) {
            return parameterGroup;
        }
        return {
            ...parameterGroup,
            parameters: parameterGroup.parameters.map((parameter) => {
                if (parameter.key !== 'subset_split' || !isParameterGroup(parameter)) {
                    return parameter;
                }
                return {
                    ...parameter,
                    parameters: parameter.parameters.map((subsetSplitParameter) => {
                        if (
                            isParameterGroup(subsetSplitParameter) ||
                            ![TRAINING_SUBSET_KEY, VALIDATION_SUBSET_KEY, TEST_SUBSET_KEY].includes(
                                subsetSplitParameter.key
                            )
                        ) {
                            return subsetSplitParameter;
                        }
                        return {
                            ...subsetSplitParameter,
                            value: getValueForKey(subsetSplitParameter),
                        } as ConfigurableParameter;
                    }),
                };
            }),
        };
    }),
});

export const TrainingSubsets = ({
    defaultSubsetParameters,
    subsetsParameters,
    onTrainingConfigurationChange,
    hasSupportedModels,
}: TrainingSubsetsProps) => {
    const { trainingSubset, validationSubset } = getSubsets(subsetsParameters);

    const areTrainingSubsetParametersChanged = !isEqual(defaultSubsetParameters, subsetsParameters);

    const [subsetsDistribution, setSubsetsDistribution] = useState<number[]>([
        trainingSubset.value,
        trainingSubset.value + validationSubset.value,
    ]);

    const trainingSubsetRatio = subsetsDistribution[0];
    const validationSubsetRatio = subsetsDistribution[1] - trainingSubsetRatio;
    const testSubsetRatio = MAX_RATIO_VALUE - subsetsDistribution[1];

    const handleUpdateSubsetsConfiguration = (values: number[]): void => {
        const trainingSubsetValue = values[0];
        const validationSubsetValue = values[1] - trainingSubsetValue;
        const testSubsetValue = MAX_RATIO_VALUE - values[1];

        const KEY_VALUE_MAP: Record<string, number> = {
            [TRAINING_SUBSET_KEY]: trainingSubsetValue,
            [VALIDATION_SUBSET_KEY]: validationSubsetValue,
            [TEST_SUBSET_KEY]: testSubsetValue,
        };

        onTrainingConfigurationChange((config) => {
            if (!config?.parameters) return undefined;
            return updateSubsetSplitValues(config, (parameter) => KEY_VALUE_MAP[parameter.key]);
        });
    };

    const handleSubsetsConfigurationReset = (): void => {
        setSubsetsDistribution([
            trainingSubset.default_value,
            trainingSubset.default_value + validationSubset.default_value,
        ]);

        onTrainingConfigurationChange((config) => {
            if (!config?.parameters) return undefined;
            return updateSubsetSplitValues(config, (parameter) => parameter.default_value as number);
        });
    };

    const { trainingSubsetSize, validationSubsetSize, testSubsetSize } = getSubsetsSizes(
        subsetsParameters,
        validationSubsetRatio,
        testSubsetRatio
    );

    const subsetsSizesInvalid = areTrainingSubsetParametersChanged && !areSubsetsSizesValid(subsetsParameters);
    const isChangedDistributionWarningVisible = hasSupportedModels && areTrainingSubsetParametersChanged;

    return (
        <Accordion>
            <Accordion.Title>
                Training subsets
                <Accordion.Tag ariaLabel={'Training subsets tag'}>
                    {trainingSubsetRatio}/{validationSubsetRatio}/{testSubsetRatio}%
                </Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content>
                <Accordion.Description>
                    Specify the distribution of annotated samples that have NOT already been assigned to a subset. Note
                    that samples used in previous training rounds already have a subset and this will remain unchanged,
                    to avoid data contamination and evaluation bias.
                </Accordion.Description>
                <Accordion.Divider marginY={'size-250'} />
                <View>
                    <SubsetsDistribution
                        subsetsDistribution={subsetsDistribution}
                        onSubsetsDistributionChange={setSubsetsDistribution}
                        testSubsetSize={testSubsetSize}
                        trainingSubsetSize={trainingSubsetSize}
                        validationSubsetSize={validationSubsetSize}
                        onSubsetsDistributionChangeEnd={handleUpdateSubsetsConfiguration}
                        onSubsetsDistributionReset={handleSubsetsConfigurationReset}
                    />
                </View>

                <Flex direction={'column'} gap={'size-200'} marginTop={'size-200'}>
                    {subsetsSizesInvalid && <TrainingSubsetsUnavailable />}
                    {isChangedDistributionWarningVisible && <TrainingSubsetsChangedDistributionWarning />}
                </Flex>
            </Accordion.Content>
        </Accordion>
    );
};
