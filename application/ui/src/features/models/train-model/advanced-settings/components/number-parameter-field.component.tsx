// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Flex, NumberField, Slider } from '@geti/ui';

import { NumberConfigurableParameter } from '../../../../../constants/shared-types';

const getDecimalPoints = (value: number): number => {
    // When log10 returns 0 (log10(1) = 0) we need to return 1
    return Math.abs(Math.ceil(Math.log10(value))) || 1;
};

const getFloatingPointStep = (minValue: number, maxValue: number): number => {
    const exponent = getDecimalPoints(maxValue - minValue);

    return 1 / Math.pow(10, exponent + 3);
};

type NumberGroupParamsProps = {
    name: string;
    onChange: (value: number) => void;
    isDisabled?: boolean;
    step?: number;
    minValue: number | null;
    maxValue: number | null;
    value: number;
    type: NumberConfigurableParameter['value_type'];
};

const DEFAULT_INT_STEP = 1;
const DEFAULT_FLOAT_STEP = 0.1;

export const getStep = ({
    step,
    maxValue,
    minValue,
    type,
}: {
    step?: number;
    minValue: number | null;
    maxValue: number | null;
    type: NumberConfigurableParameter['value_type'];
}): number => {
    if (step !== undefined) {
        return step;
    }

    if (maxValue === null || minValue === null) {
        return type === 'int' ? DEFAULT_INT_STEP : DEFAULT_FLOAT_STEP;
    }

    return getFloatingPointStep(minValue, maxValue);
};

export const NumberParameterField = ({
    value,
    minValue,
    maxValue,
    type,
    onChange,
    isDisabled,
    name,
    step,
}: NumberGroupParamsProps) => {
    const [draftValue, setDraftValue] = useState<number | null>(null);
    const parameterValue = draftValue ?? value;

    const fieldStep = getStep({ step, type, maxValue, minValue });
    const formatOptions = type === 'float' ? { maximumFractionDigits: Math.abs(Math.log10(fieldStep)) } : undefined;

    const handleValueChange = (inputValue: number): void => {
        setDraftValue(inputValue);
        onChange(inputValue);
    };

    if (maxValue === null || minValue === null) {
        return (
            <NumberField
                aria-label={`Change ${name}`}
                step={fieldStep}
                value={parameterValue}
                minValue={minValue === null ? undefined : minValue}
                maxValue={maxValue === null ? undefined : maxValue}
                onChange={onChange}
                isDisabled={isDisabled}
            />
        );
    }

    return (
        <Flex gap={'size-100'}>
            <Slider
                aria-label={`Change ${name} slider`}
                value={parameterValue}
                minValue={minValue}
                maxValue={maxValue}
                onChange={setDraftValue}
                onChangeEnd={handleValueChange}
                step={fieldStep}
                isFilled
                flex={1}
                isDisabled={isDisabled}
            />
            <NumberField
                isQuiet
                step={fieldStep}
                value={parameterValue}
                minValue={minValue}
                maxValue={maxValue}
                onChange={handleValueChange}
                isDisabled={isDisabled}
                aria-label={`Change ${name}`}
                formatOptions={formatOptions}
            />
        </Flex>
    );
};
