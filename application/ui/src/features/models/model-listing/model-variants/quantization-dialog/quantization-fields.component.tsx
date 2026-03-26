// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { ActionButton, Checkbox, Content, ContextualHelp, Grid, Text } from '@geti/ui';
import { Refresh } from '@geti/ui/icons';

import { NumberParameterField } from '../../../train-model/advanced-settings/components/number-parameter-field.component';

const DEFAULT_QUANTIZATION_PARAMETERS = {
    accuracyDrop: 3.0,
    calibrationSize: 200,
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

export const MaxAccuracyDropField = () => {
    const [accuracyDrop, setAccuracyDrop] = useState<number>(DEFAULT_QUANTIZATION_PARAMETERS.accuracyDrop);
    const [hasNoMaxAccuracyDrop, setHasNoMaxAccuracyDrop] = useState<boolean>(false);

    return (
        <QuantizationFieldLayout
            onReset={() => {
                setAccuracyDrop(DEFAULT_QUANTIZATION_PARAMETERS.accuracyDrop);
            }}
        >
            <Text>Max accuracy drop</Text>
            <ContextualHelp>
                <Content>Maximum allowed drop in validation accuracy</Content>
            </ContextualHelp>
            <NumberParameterField
                name='Max accuracy drop'
                value={accuracyDrop}
                minValue={0.1}
                maxValue={10.0}
                type={'float'}
                isDisabled={hasNoMaxAccuracyDrop}
                onChange={setAccuracyDrop}
                step={0.1}
            />
            <Checkbox aria-label='No maximum' isSelected={hasNoMaxAccuracyDrop} onChange={setHasNoMaxAccuracyDrop}>
                No maximum
            </Checkbox>
        </QuantizationFieldLayout>
    );
};

export const CalibrationDatasetSizeField = () => {
    const [calibrationSize, setCalibrationSize] = useState<number>(DEFAULT_QUANTIZATION_PARAMETERS.calibrationSize);
    const [usesFullCalibrationDataset, setUsesFullCalibrationDataset] = useState<boolean>(false);

    return (
        <QuantizationFieldLayout
            onReset={() => {
                setCalibrationSize(DEFAULT_QUANTIZATION_PARAMETERS.calibrationSize);
            }}
        >
            <Text>Max calibration size</Text>
            <ContextualHelp>
                <Content>Calibration samples will be randomly selected within the validation set</Content>
            </ContextualHelp>
            <NumberParameterField
                name='Max calibration size'
                value={calibrationSize}
                minValue={1}
                maxValue={1000}
                isDisabled={usesFullCalibrationDataset}
                type={'int'}
                onChange={setCalibrationSize}
                step={1}
            />
            <Checkbox
                aria-label='Use full dataset'
                isSelected={usesFullCalibrationDataset}
                onChange={setUsesFullCalibrationDataset}
            >
                Use full dataset
            </Checkbox>
        </QuantizationFieldLayout>
    );
};
