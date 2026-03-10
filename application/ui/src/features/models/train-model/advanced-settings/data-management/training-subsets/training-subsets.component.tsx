// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, useState } from 'react';

import { Content, Flex, Grid, Heading, InlineAlert, minmax, Text, View } from '@geti/ui';
import { useGetDatasetItems } from 'hooks/use-get-dataset-items.hook';
import { isEqual } from 'lodash-es';

import type { ConfigurableParameter, TrainingConfiguration } from '../../../../../../constants/shared-types';
import { isParameterGroup } from '../../../../model-listing/model-training-parameters/utils';
import { Accordion } from '../../components/accordion/accordion.component';
import { ResetButton } from '../../components/reset-button.component';
import { ResultingDatasetDistribution } from './resulting-dataset-distribution.component';
import { SubsetDistributionStats } from './subset-distribution-stats.component';
import { SubsetsDistributionSlider } from './subsets-distribution-slider/subsets-distribution-slider.component';
import {
    areSubsetsSizesValid,
    getSubsets,
    MAX_RATIO_VALUE,
    SubsetSplitParameters,
    TEST_SUBSET_KEY,
    TRAINING_SUBSET_KEY,
    VALIDATION_SUBSET_KEY,
} from './utils';

import classes from './training-subsets.module.scss';

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
                areas={['label label', 'slider reset', 'counts counts']}
                columns={[minmax('size-3400', '1fr'), 'max-content']}
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
                    label={'Distribution for new samples'}
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
                    totalSize={trainingSubsetSize + validationSubsetSize + testSubsetSize}
                />
            </Grid>
        </View>
    );
};

type TrainingSubsetsProps = {
    defaultSubsetParameters: SubsetSplitParameters;
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

const useSubsetDatasetSizes = () => {
    const { data: trainingDatasetItems } = useGetDatasetItems({
        limit: 1,
        annotationStatus: 'reviewed',
        subset: 'training',
    });
    const { data: testingDatasetItems } = useGetDatasetItems({
        limit: 1,
        annotationStatus: 'reviewed',
        subset: 'testing',
    });
    const { data: validationDatasetItems } = useGetDatasetItems({
        limit: 1,
        annotationStatus: 'reviewed',
        subset: 'validation',
    });
    const { data: unassignedDatasetItems } = useGetDatasetItems({
        limit: 1,
        annotationStatus: 'reviewed',
        subset: 'unassigned',
    });

    const trainingSubsetSize = trainingDatasetItems?.pagination?.total ?? 0;
    const testingSubsetSize = testingDatasetItems?.pagination?.total ?? 0;
    const validationSubsetSize = validationDatasetItems?.pagination?.total ?? 0;
    const unassignedSubsetSize = unassignedDatasetItems?.pagination?.total ?? 0;

    const assignedDatasetItemsSize = trainingSubsetSize + testingSubsetSize + validationSubsetSize;
    const totalDatasetItemsSize = assignedDatasetItemsSize + unassignedSubsetSize;

    return {
        trainingSubsetSize,
        testingSubsetSize,
        validationSubsetSize,
        unassignedSubsetSize,
        assignedDatasetItemsSize,
        totalDatasetItemsSize,
    };
};

export const TrainingSubsets = ({
    defaultSubsetParameters,
    subsetsParameters,
    onTrainingConfigurationChange,
}: TrainingSubsetsProps) => {
    const { trainingSubset, validationSubset } = getSubsets(subsetsParameters);
    const {
        validationSubsetSize,
        testingSubsetSize,
        unassignedSubsetSize,
        assignedDatasetItemsSize,
        totalDatasetItemsSize,
        trainingSubsetSize,
    } = useSubsetDatasetSizes();

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

    const newValidationSubsetSize = Math.floor((validationSubsetRatio / 100) * unassignedSubsetSize);
    const newTestingSubsetSize = Math.floor((testSubsetRatio / 100) * unassignedSubsetSize);
    const newTrainingSubsetSize = unassignedSubsetSize - newValidationSubsetSize - newTestingSubsetSize;

    const areSubsetsSizesValid = () => {
        return ![newValidationSubsetSize, newTestingSubsetSize, newTrainingSubsetSize].some((size) => size === 0);
    };

    const subsetsSizesInvalid = areTrainingSubsetParametersChanged && !areSubsetsSizesValid();

    return (
        <Accordion>
            <Accordion.Title>
                Training subsets
                <Accordion.Tag ariaLabel={'Training subsets tag'}>
                    {trainingSubsetRatio}/{validationSubsetRatio}/{testSubsetRatio}%
                </Accordion.Tag>
            </Accordion.Title>
            <Accordion.Content UNSAFE_className={classes.trainingSubsets}>
                <Accordion.Description>
                    Specify the distribution of annotated samples that have NOT already been assigned to a subset. Note
                    that samples used in previous training rounds already have a subset and this will remain unchanged,
                    to avoid data contamination and evaluation bias.
                </Accordion.Description>
                <Accordion.Divider marginY={'size-200'} />
                <View>
                    <Text>Dataset: {totalDatasetItemsSize} samples</Text>
                    <Flex alignItems={'center'} gap={'size-100'}>
                        <Text>Assigned: {assignedDatasetItemsSize}</Text>
                        <Text>Unassigned: {unassignedSubsetSize}</Text>
                    </Flex>
                    <Accordion.Divider marginY={'size-200'} />
                    <SubsetsDistribution
                        subsetsDistribution={subsetsDistribution}
                        onSubsetsDistributionChange={setSubsetsDistribution}
                        testSubsetSize={newTestingSubsetSize}
                        trainingSubsetSize={newTrainingSubsetSize}
                        validationSubsetSize={newValidationSubsetSize}
                        onSubsetsDistributionChangeEnd={handleUpdateSubsetsConfiguration}
                        onSubsetsDistributionReset={handleSubsetsConfigurationReset}
                    />
                    <Accordion.Divider marginY={'size-200'} />
                    <ResultingDatasetDistribution
                        trainingSubsetSize={trainingSubsetSize}
                        validationSubsetSize={validationSubsetSize}
                        testingSubsetSize={testingSubsetSize}
                        newTrainingSubsetSize={newTrainingSubsetSize}
                        newValidationSubsetSize={newValidationSubsetSize}
                        newTestingSubsetSize={newTestingSubsetSize}
                        totalDatasetItemsSize={totalDatasetItemsSize}
                    />
                </View>

                <Flex direction={'column'} gap={'size-200'} marginTop={'size-200'}>
                    {subsetsSizesInvalid && <TrainingSubsetsUnavailable />}
                </Flex>
            </Accordion.Content>
        </Accordion>
    );
};
