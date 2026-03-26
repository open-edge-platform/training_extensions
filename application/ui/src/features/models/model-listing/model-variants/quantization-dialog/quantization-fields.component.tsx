// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { ActionButton, Checkbox, Content, ContextualHelp, Grid, Text } from '@geti/ui';
import { Refresh } from '@geti/ui/icons';

import { NumberParameterField } from '../../../../../components/fields/number-parameter-field/number-parameter-field.component';

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

interface MaxAccuracyDropFieldProps {
    value: number;
    onChange: (value: number) => void;
    isDisabled: boolean;
    onDisabledChange: (isDisabled: boolean) => void;
    onReset: () => void;
}

export const MaxAccuracyDropField = ({
    value,
    onChange,
    isDisabled,
    onDisabledChange,
    onReset,
}: MaxAccuracyDropFieldProps) => {
    return (
        <QuantizationFieldLayout onReset={onReset}>
            <Text>Max accuracy drop (%)</Text>
            <ContextualHelp>
                <Content>Maximum allowed drop in validation accuracy</Content>
            </ContextualHelp>
            <NumberParameterField
                name='Max accuracy drop'
                value={value}
                minValue={0.1}
                maxValue={15}
                type={'float'}
                isDisabled={isDisabled}
                onChange={onChange}
                step={0.1}
            />
            <Checkbox aria-label='No maximum' isSelected={isDisabled} onChange={onDisabledChange}>
                No maximum
            </Checkbox>
        </QuantizationFieldLayout>
    );
};

interface CalibrationDatasetSizeFieldProps {
    value: number;
    onChange: (value: number) => void;
    maxValue: number;
    isDisabled: boolean;
    onDisabledChange: (isDisabled: boolean) => void;
    onReset: () => void;
}

export const CalibrationDatasetSizeField = ({
    value,
    onChange,
    maxValue,
    isDisabled,
    onDisabledChange,
    onReset,
}: CalibrationDatasetSizeFieldProps) => {
    return (
        <QuantizationFieldLayout onReset={onReset}>
            <Text>Max calibration size</Text>
            <ContextualHelp>
                <Content>Calibration samples will be randomly selected within the dataset</Content>
            </ContextualHelp>
            <NumberParameterField
                name='Max calibration size'
                value={value}
                minValue={1}
                maxValue={maxValue}
                isDisabled={isDisabled}
                type={'int'}
                onChange={onChange}
                step={1}
            />
            <Checkbox aria-label='Use full dataset' isSelected={isDisabled} onChange={onDisabledChange}>
                Use full dataset
            </Checkbox>
        </QuantizationFieldLayout>
    );
};
