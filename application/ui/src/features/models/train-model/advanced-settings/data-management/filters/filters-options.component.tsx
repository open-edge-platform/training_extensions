// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FC } from 'react';

import { ActionButton, Checkbox, Flex, NumberField, Text } from '@geti/ui';
import { Refresh } from '@geti/ui/icons';

import { BoolParameter, NumberParameter } from '../../../../configuration.interface';
import { Tooltip } from '../../ui/tooltip.component';

export type FilterOption = [BoolParameter, NumberParameter];

interface FilterOptionProps {
    option: [BoolParameter, NumberParameter];
    onOptionChange: (option: FilterOption) => void;
}

interface FilterOptionTooltipProps {
    description: string;
}

const FilterOptionTooltip: FC<FilterOptionTooltipProps> = ({ description }) => {
    return <Tooltip>{description}</Tooltip>;
};

const FilterOption: FC<FilterOptionProps> = ({ option, onOptionChange }) => {
    const [enableParameter, configParameter] = option;
    const { name, description, min_value, max_value, value } = configParameter;
    const isUnlimited = !enableParameter.value;

    const handleRest = () => {
        onOptionChange([
            { ...enableParameter, value: enableParameter.default_value },
            { ...configParameter, value: configParameter.default_value },
        ]);
    };

    const handleFilterValueChange = (inputValue: number) => {
        const updatedOption: FilterOption = [enableParameter, { ...configParameter, value: inputValue }];
        onOptionChange(updatedOption);
    };

    const handleUnlimitedChange = (inputValue: boolean) => {
        const updatedOption: FilterOption = [{ ...enableParameter, value: !inputValue }, configParameter];
        onOptionChange(updatedOption);
    };

    return (
        <>
            <Text gridColumn={'1/2'}>
                {name} <FilterOptionTooltip description={description} />
            </Text>
            <Flex gap={'size-200'} gridColumn={'2/3'}>
                <NumberField
                    aria-label={`Change ${name}`}
                    minValue={min_value ?? undefined}
                    maxValue={max_value ?? undefined}
                    step={1}
                    value={value}
                    isDisabled={isUnlimited}
                    onChange={handleFilterValueChange}
                />

                <Checkbox
                    isEmphasized
                    isSelected={isUnlimited}
                    onChange={handleUnlimitedChange}
                    aria-label={`Toggle ${configParameter.name}`}
                >
                    Unlimited
                </Checkbox>
            </Flex>
            <ActionButton gridColumn={'3/4'} isQuiet aria-label={`Reset ${name}`} onPress={handleRest}>
                <Refresh />
            </ActionButton>
        </>
    );
};

type FiltersOptionsProps = {
    options: FilterOption[];
    onOptionsChange: (option: FilterOption) => void;
};

export const FiltersOptions = ({ options, onOptionsChange }: FiltersOptionsProps) => {
    return (
        <>
            {options.map((option) => (
                <FilterOption
                    key={`${option.at(0)?.key}-${option.at(1)?.key}`}
                    option={option}
                    onOptionChange={onOptionsChange}
                />
            ))}
        </>
    );
};
