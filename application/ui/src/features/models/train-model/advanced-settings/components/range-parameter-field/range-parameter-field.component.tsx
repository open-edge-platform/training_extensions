// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Flex, NumberField, RangeSlider } from '@geti-ui/ui';
import type { RangeValue } from '@react-types/shared';

import { FloatConfigurableRangeParameter } from '../../../../../../constants/shared-types';
import { getFloatingPointStep } from '../../utils';

import classes from './range-parameter-field.module.scss';

type RangeParameterFieldProps = {
    onChange: (value: [number, number]) => void;
    isDisabled?: boolean;
    step?: number;
    name: string;
    value: FloatConfigurableRangeParameter['value'];
    defaultValue: FloatConfigurableRangeParameter['default_value'];
};

const getStep = ({ step, maxValue, minValue }: { step?: number; minValue: number; maxValue: number }): number => {
    if (step !== undefined) {
        return step;
    }

    return getFloatingPointStep(minValue, maxValue);
};

export const RangeParameterField = ({
    defaultValue,
    value,
    onChange,
    isDisabled,
    name,
    step,
}: RangeParameterFieldProps) => {
    const [draftRange, setDraftRange] = useState<RangeValue<number> | null>(null);
    const parameterValue = draftRange ?? { start: value[0], end: value[1] };

    const fieldStep = getStep({ step, maxValue: defaultValue[1], minValue: defaultValue[0] });
    const decimalPlaces = (fieldStep.toString().split('.')[1] || '').length;

    const handleRangeChangeEnd = (inputValue: RangeValue<number>): void => {
        const { start, end } = inputValue;

        setDraftRange(null);
        onChange([start, end]);
    };

    const handleRangeChange = (inputValue: RangeValue<number>): void => {
        const { start, end } = inputValue;
        // Prevent start and end from being equal
        if (end - start >= fieldStep) {
            setDraftRange({ start, end });
        }
    };

    const handleNumberChange = (start: number, end: number): void => {
        // Prevent start and end from being equal
        if (end - start >= fieldStep) {
            setDraftRange(null);
            onChange([start, end]);
        }
    };

    return (
        <Flex gap={'size-100'}>
            <NumberField
                isQuiet
                step={fieldStep}
                value={parameterValue.start}
                minValue={defaultValue[0]}
                maxValue={defaultValue[1]}
                onChange={(start) => handleNumberChange(start, parameterValue.end)}
                isDisabled={isDisabled}
                aria-label={`Change ${name} start range value`}
                formatOptions={{ maximumFractionDigits: decimalPlaces }}
            />
            <RangeSlider
                value={parameterValue}
                minValue={defaultValue[0]}
                maxValue={defaultValue[1]}
                defaultValue={{ start: defaultValue[0], end: defaultValue[1] }}
                onChange={handleRangeChange}
                onChangeEnd={handleRangeChangeEnd}
                step={fieldStep}
                flex={1}
                isDisabled={isDisabled}
                aria-label={`Change ${name} value`}
                UNSAFE_className={isDisabled ? '' : classes.rangeSlider}
            />
            <NumberField
                isQuiet
                step={fieldStep}
                value={parameterValue.end}
                minValue={defaultValue[0]}
                maxValue={defaultValue[1]}
                onChange={(end) => handleNumberChange(parameterValue.start, end)}
                isDisabled={isDisabled}
                aria-label={`Change ${name} end range value`}
                formatOptions={{ maximumFractionDigits: decimalPlaces }}
            />
        </Flex>
    );
};
