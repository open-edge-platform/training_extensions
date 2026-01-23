// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FC, ReactNode } from 'react';

import { Grid, Item, minmax, Picker, Text, ToggleButtons, View } from '@geti/ui';
import { isArray, isBoolean, isFunction } from 'lodash-es';

import { ConfigurationParameter, EnumConfigurationParameter } from '../../../configuration.interface';
import { isBoolEnableParameter } from '../utils';
import { BooleanParameter } from './boolean-parameter.component';
import { NumberParameterField } from './number-parameter-field.component';
import { RangeParameterField } from './range-parameter-field.component';
import { ResetButton } from './reset-button.component';
import { Tooltip } from './tooltip.component';

type ParametersProps = {
    parameters: ConfigurationParameter[];
    onChange: (parameter: ConfigurationParameter) => void;
    isReadOnly?: boolean;
};

const ParameterTooltip: FC<{ text: string }> = ({ text }) => {
    return <Tooltip>{text}</Tooltip>;
};

type ParameterProps = {
    parameter: ConfigurationParameter;
    onChange: (parameter: ConfigurationParameter) => void;
    isDisabled?: boolean;
    marginStart?: string;
    isReadOnly: boolean;
};

type ParameterFieldProps = {
    parameter: ConfigurationParameter;
    onChange: (parameter: ConfigurationParameter) => void;
    isDisabled?: boolean;
};

type ParameterLayoutProps = {
    header: string;
    description: string;
    onReset?: () => void;
    children: ReactNode;
    marginStart?: string;
};

type ParameterNameProps = {
    name: string;
    description: string;
    gridColumn?: string;
    marginStart?: string;
};

export const ParameterName = ({ name, description, marginStart, gridColumn }: ParameterNameProps) => {
    return (
        <Text marginStart={marginStart} gridColumn={gridColumn}>
            {name}
            <ParameterTooltip text={description} />
        </Text>
    );
};

const ParameterLayout: FC<ParameterLayoutProps> = ({ header, children, description, onReset, marginStart }) => {
    return (
        <>
            <ParameterName name={header} description={description} gridColumn={'1/2'} marginStart={marginStart} />
            <View gridColumn={'2/3'}>{children}</View>
            {isFunction(onReset) && <ResetButton onPress={onReset} aria-label={`Reset ${header}`} />}
        </>
    );
};

type ParameterReadOnlyProps = {
    parameter: Pick<ConfigurationParameter, 'value' | 'name' | 'description'>;
    marginStart?: string;
};

type ParameterReadOnlyValueProps = Pick<ConfigurationParameter, 'value' | 'name'>;

export const ParameterReadOnlyValue = ({ value, name }: ParameterReadOnlyValueProps) => {
    if (isBoolean(value)) {
        return <span aria-label={name}>{value ? 'On' : 'Off'}</span>;
    }

    if (isArray(value) && value.length === 2) {
        return (
            <span aria-label={name}>
                {value[0]} - {value[1]}
            </span>
        );
    }

    return <span aria-label={name}>{value}</span>;
};

const ParameterReadOnly = ({ parameter, marginStart }: ParameterReadOnlyProps) => {
    return (
        <ParameterLayout header={parameter.name} description={parameter.description} marginStart={marginStart}>
            <ParameterReadOnlyValue value={parameter.value} name={parameter.name} />
        </ParameterLayout>
    );
};

export const EnumParameterField = ({
    parameter,
    onChange,
    isDisabled,
}: {
    parameter: EnumConfigurationParameter;
    onChange: (parameter: EnumConfigurationParameter) => void;
    isDisabled?: boolean;
}) => {
    const handleChange = (value: EnumConfigurationParameter['value']) => {
        onChange({
            ...parameter,
            value,
        });
    };

    if (parameter.allowed_values.length < 4) {
        return (
            <ToggleButtons
                options={parameter.allowed_values}
                selectedOption={parameter.value}
                onOptionChange={handleChange}
                isDisabled={isDisabled}
            />
        );
    }

    const items = parameter.allowed_values.map((value) => ({ value }));

    return (
        <Picker
            items={items}
            selectedKey={parameter.value.toString()}
            onSelectionChange={(key) => handleChange(key as EnumConfigurationParameter['value'])}
            aria-label={`Select ${parameter.name}`}
        >
            {(item) => (
                <Item key={item.value}>
                    <Text>{item.value}</Text>
                </Item>
            )}
        </Picker>
    );
};

const ParameterField: FC<ParameterFieldProps> = ({ parameter, onChange, isDisabled }) => {
    if (parameter.type === 'enum') {
        return <EnumParameterField parameter={parameter} onChange={onChange} />;
    }

    if (parameter.type === 'float' || parameter.type === 'int') {
        const handleChange = (value: number) => {
            onChange({
                ...parameter,
                value,
            });
        };

        return (
            <NumberParameterField
                value={parameter.value}
                minValue={parameter.min_value}
                maxValue={parameter.max_value}
                onChange={handleChange}
                type={parameter.type}
                isDisabled={isDisabled}
                name={parameter.name}
            />
        );
    }

    if (parameter.type === 'array') {
        const handleChange = (value: number[]) => {
            onChange({
                ...parameter,
                value,
            });
        };

        return (
            <RangeParameterField
                value={parameter.value}
                defaultValue={parameter.default_value}
                onChange={handleChange}
                isDisabled={isDisabled}
                name={parameter.name}
            />
        );
    }

    if (parameter.type === 'bool') {
        const handleChange = (value: boolean) => {
            onChange({
                ...parameter,
                value,
            });
        };

        return (
            <BooleanParameter
                value={parameter.value}
                header={parameter.name}
                onChange={handleChange}
                isDisabled={isDisabled}
            />
        );
    }
};

export const Parameter = ({ parameter, onChange, isDisabled, marginStart, isReadOnly }: ParameterProps) => {
    if (isReadOnly) {
        return <ParameterReadOnly parameter={parameter} marginStart={marginStart} />;
    }

    const handleReset = () => {
        onChange({ ...parameter, value: parameter.default_value } as ConfigurationParameter);
    };

    return (
        <ParameterLayout
            header={parameter.name}
            description={parameter.description}
            onReset={handleReset}
            marginStart={marginStart}
        >
            <ParameterField parameter={parameter} onChange={onChange} isDisabled={isDisabled} />
        </ParameterLayout>
    );
};

type ParametersListProps = {
    parameters: ConfigurationParameter[];
    onChange: (parameter: ConfigurationParameter) => void;
    isReadOnly: boolean;
};

const ParametersList = ({ parameters, onChange, isReadOnly }: ParametersListProps) => {
    if (isBoolEnableParameter(parameters[0])) {
        return parameters.map((parameter, index) => (
            <Parameter
                key={parameter.name}
                parameter={parameter}
                onChange={onChange}
                isDisabled={index > 0 && !parameters[0].value}
                marginStart={index > 0 ? 'size-150' : undefined}
                isReadOnly={isReadOnly}
            />
        ));
    }

    return parameters.map((parameter) => (
        <Parameter key={parameter.name} parameter={parameter} onChange={onChange} isReadOnly={isReadOnly} />
    ));
};

const ParametersContainer = ({ children, isReadOnly }: { children: ReactNode; isReadOnly?: boolean }) => {
    const columns = isReadOnly ? ['size-3000', '1fr'] : ['size-3000', minmax('size-3400', '1fr'), 'size-400'];

    return (
        <Grid columns={columns} gap={'size-300'} alignItems={'center'}>
            {children}
        </Grid>
    );
};

export const Parameters = ({ parameters, onChange, isReadOnly = false }: ParametersProps) => {
    return (
        <ParametersContainer>
            <ParametersList parameters={parameters} onChange={onChange} isReadOnly={isReadOnly} />
        </ParametersContainer>
    );
};

Parameters.Container = ParametersContainer;
Parameter.Layout = ParameterLayout;
