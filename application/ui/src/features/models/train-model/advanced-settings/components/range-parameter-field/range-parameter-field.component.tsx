// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState } from 'react';

import { Flex, NumberField, RangeSlider, type RangeValue } from '@geti-ui/ui';
import { isEqual } from 'lodash-es';

import { FloatConfigurableRangeParameter } from '../../../../../../constants/shared-types';
import { getStep } from '../utils';

import classes from './range-parameter-field.module.scss';

type RangeParameterValue = FloatConfigurableRangeParameter['value'];

type RangeParameterFieldProps = {
    onChange: (value: RangeParameterValue) => void;
    isDisabled?: boolean;
    step?: number;
    name: string;
    value: RangeParameterValue;
    maxValue: number;
    minValue: number;
};

export const RangeParameterField = ({
    value,
    onChange,
    isDisabled,
    name,
    step,
    minValue,
    maxValue,
}: RangeParameterFieldProps) => {
    const [parameterValues, setParameterValues] = useState<RangeValue<number>>({ start: value[0], end: value[1] });
    const prevValues = useRef<RangeParameterValue>(value);

    if (!isEqual(prevValues.current, value)) {
        prevValues.current = value;
        setParameterValues({ start: value[0], end: value[1] });
    }

    const fieldStep = getStep({ step, type: 'float', maxValue, minValue });
    const decimalPlaces = (fieldStep.toString().split('.')[1] || '').length;

    const handleRangeChangeEnd = (inputValue: RangeValue<number>): void => {
        const { start, end } = inputValue;

        setParameterValues(inputValue);
        onChange([start, end]);
    };

    const handleNumberChange = (start: number, end: number): void => {
        setParameterValues({ start, end });
        onChange([start, end]);
    };

    return (
        <Flex gap={'size-100'}>
            <NumberField
                step={fieldStep}
                hideStepper
                width={'size-900'}
                value={parameterValues.start}
                minValue={minValue}
                maxValue={parameterValues.end}
                onChange={(start) => handleNumberChange(start, parameterValues.end)}
                isDisabled={isDisabled}
                aria-label={`Change ${name} start range value`}
                formatOptions={{ maximumFractionDigits: decimalPlaces }}
            />
            <RangeSlider
                minValue={minValue}
                maxValue={maxValue}
                value={parameterValues}
                onChange={setParameterValues}
                onChangeEnd={handleRangeChangeEnd}
                step={fieldStep}
                flex={1}
                isDisabled={isDisabled}
                aria-label={`Change ${name} value`}
                UNSAFE_className={isDisabled ? '' : classes.rangeSlider}
            />
            <NumberField
                hideStepper
                width={'size-900'}
                step={fieldStep}
                value={parameterValues.end}
                minValue={parameterValues.start}
                maxValue={maxValue}
                onChange={(end) => handleNumberChange(parameterValues.start, end)}
                isDisabled={isDisabled}
                aria-label={`Change ${name} end range value`}
                formatOptions={{ maximumFractionDigits: decimalPlaces }}
            />
        </Flex>
    );
};
