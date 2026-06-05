// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { ActionButton, Checkbox, Content, ContextualHelp, Flex, Grid, NumberField, Slider, Text } from '@geti/ui';
import { Refresh } from '@geti/ui/icons';

export const DEFAULT_QUANTIZATION_PARAMETERS = {
    accuracyDrop: 1.0,
    calibrationSize: 200,
    hasNoMaxAccuracyDrop: true,
    usesFullCalibrationDataset: false,
};

type QuantizationFieldLayoutProps = {
    children: ReactNode;
    onReset: () => void;
};
const QuantizationFieldLayout = ({ children, onReset }: QuantizationFieldLayoutProps) => (
    <Grid columns={['1fr', '.1fr', 'size-3400', '1fr', '.2fr']} alignItems={'center'} gap={'size-200'}>
        {children}

        <ActionButton isQuiet aria-label={'Reset to default'} onPress={onReset}>
            <Refresh />
        </ActionButton>
    </Grid>
);

type MaxAccuracyDropFieldProps = {
    value: number;
    onChange: (value: number) => void;
    isDisabled: boolean;
    onDisabledChange: (isDisabled: boolean) => void;
    onReset: () => void;
};

export const MaxAccuracyDropField = ({
    value,
    onChange,
    isDisabled,
    onDisabledChange,
    onReset,
}: MaxAccuracyDropFieldProps) => {
    const [draftValue, setDraftValue] = useState<number | null>(null);
    const parameterValue = draftValue ?? value;

    const handleValueChange = (inputValue: number) => {
        setDraftValue(null);
        onChange(inputValue);
    };

    return (
        <QuantizationFieldLayout onReset={onReset}>
            <Text>Max accuracy drop (%)</Text>
            <ContextualHelp>
                <Content>Maximum allowed drop in validation accuracy</Content>
            </ContextualHelp>
            <Flex gap={'size-100'}>
                <Slider
                    aria-label={'Change Max accuracy drop slider'}
                    value={parameterValue}
                    minValue={0.1}
                    maxValue={15}
                    step={0.1}
                    onChange={setDraftValue}
                    onChangeEnd={handleValueChange}
                    isFilled
                    flex={1}
                    isDisabled={isDisabled}
                />
                <NumberField
                    hideStepper
                    step={0.1}
                    value={parameterValue}
                    minValue={0.1}
                    maxValue={15}
                    onChange={handleValueChange}
                    isDisabled={isDisabled}
                    aria-label={'Change Max accuracy drop'}
                    formatOptions={{ maximumFractionDigits: 1 }}
                />
            </Flex>
            <Checkbox aria-label='No maximum' isSelected={isDisabled} onChange={onDisabledChange}>
                No maximum
            </Checkbox>
        </QuantizationFieldLayout>
    );
};

type CalibrationDatasetSizeFieldProps = {
    value: number;
    onChange: (value: number) => void;
    maxValue: number;
    isDisabled: boolean;
    onDisabledChange: (isDisabled: boolean) => void;
    onReset: () => void;
};

export const CalibrationDatasetSizeField = ({
    value,
    onChange,
    maxValue,
    isDisabled,
    onDisabledChange,
    onReset,
}: CalibrationDatasetSizeFieldProps) => {
    const [draftValue, setDraftValue] = useState<number | null>(null);
    const parameterValue = draftValue ?? value;

    const handleValueChange = (inputValue: number) => {
        setDraftValue(null);
        onChange(inputValue);
    };

    return (
        <QuantizationFieldLayout onReset={onReset}>
            <Text>Max calibration size</Text>

            <ContextualHelp>
                <Content>Calibration samples will be randomly selected within the dataset</Content>
            </ContextualHelp>

            <Flex gap={'size-100'}>
                <Slider
                    aria-label={'Change Max calibration size slider'}
                    value={parameterValue}
                    minValue={1}
                    maxValue={maxValue}
                    step={1}
                    onChange={setDraftValue}
                    onChangeEnd={handleValueChange}
                    isFilled
                    flex={1}
                    isDisabled={isDisabled}
                />
                <NumberField
                    hideStepper
                    step={1}
                    value={parameterValue}
                    minValue={1}
                    maxValue={maxValue}
                    onChange={handleValueChange}
                    isDisabled={isDisabled}
                    aria-label={'Change Max calibration size'}
                />
            </Flex>
            <Checkbox aria-label='Use full dataset' isSelected={isDisabled} onChange={onDisabledChange}>
                Use full dataset
            </Checkbox>
        </QuantizationFieldLayout>
    );
};
