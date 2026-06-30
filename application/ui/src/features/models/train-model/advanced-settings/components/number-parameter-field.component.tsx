// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState } from 'react';

import { Flex, NumberField, Slider } from '@geti-ui/ui';

import { NumberConfigurableParameter } from '../../../../../constants/shared-types';
import { getStep } from './utils';

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
    const [parameterValue, setParameterValue] = useState<number>(value);
    const previousValueRef = useRef<number>(value);

    if (previousValueRef.current !== value) {
        previousValueRef.current = value;
        setParameterValue(value);
    }

    const fieldStep = getStep({ step, type, maxValue, minValue });
    // Preserve full precision for float values instead of rounding to the
    // NumberField default of 3 fraction digits.
    const formatOptions = type === 'float' ? { maximumFractionDigits: 20 } : undefined;
    const numberFieldStep = type === 'int' ? fieldStep : undefined;

    const handleValueChange = (inputValue: number): void => {
        setParameterValue(inputValue);
        onChange(inputValue);
    };

    if (maxValue === null || minValue === null) {
        return (
            <NumberField
                aria-label={`Change ${name}`}
                hideStepper
                width={'size-900'}
                value={parameterValue}
                minValue={minValue === null ? undefined : minValue}
                maxValue={maxValue === null ? undefined : maxValue}
                onChange={onChange}
                isDisabled={isDisabled}
                formatOptions={formatOptions}
                step={numberFieldStep}
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
                onChange={setParameterValue}
                onChangeEnd={onChange}
                step={fieldStep}
                isFilled
                flex={1}
                isDisabled={isDisabled}
            />
            <NumberField
                hideStepper
                width={'size-900'}
                value={parameterValue}
                minValue={minValue}
                maxValue={maxValue}
                onChange={handleValueChange}
                isDisabled={isDisabled}
                aria-label={`Change ${name}`}
                formatOptions={formatOptions}
                step={numberFieldStep}
            />
        </Flex>
    );
};
