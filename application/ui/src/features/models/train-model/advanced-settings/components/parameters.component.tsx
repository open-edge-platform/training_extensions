// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Content, ContextualHelp, Grid, Item, minmax, Picker, Text, ToggleButtons, View } from '@geti/ui';
import { isBoolean, isFunction } from 'lodash-es';

import type { ConfigurableParameter, NumberEnumConfigurableParameter } from '../../../../../constants/shared-types';
import { isBoolEnableParameter } from '../utils';
import { BooleanParameterField } from './boolean-parameter-field.component';
import { NumberParameterField } from './number-parameter-field.component';
import { RangeParameterField } from './range-parameter-field/range-parameter-field.component';
import { ResetButton } from './reset-button.component';

type ParametersProps = {
    parameters: ConfigurableParameter[];
    onChange: (parameter: ConfigurableParameter) => void;
    isReadOnly?: boolean;
};

const ParameterTooltip = ({ text }: { text: string }) => {
    return (
        <ContextualHelp variant='info'>
            <Content>
                <Text>{text}</Text>
            </Content>
        </ContextualHelp>
    );
};

type ParameterProps = {
    parameter: ConfigurableParameter;
    onChange: (parameter: ConfigurableParameter) => void;
    isDisabled?: boolean;
    marginStart?: string;
    isReadOnly: boolean;
};

type ParameterFieldProps = {
    parameter: ConfigurableParameter;
    onChange: (parameter: ConfigurableParameter) => void;
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

const ParameterLayout = ({ header, children, description, onReset, marginStart }: ParameterLayoutProps) => {
    return (
        <>
            <ParameterName name={header} description={description} gridColumn={'1/2'} marginStart={marginStart} />
            <View gridColumn={'2/3'}>{children}</View>
            {isFunction(onReset) && <ResetButton onPress={onReset} aria-label={`Reset ${header}`} />}
        </>
    );
};

type ParameterReadOnlyProps = {
    parameter: Pick<ConfigurableParameter, 'value' | 'name' | 'description'>;
    marginStart?: string;
};

type ParameterReadOnlyValueProps = Pick<ConfigurableParameter, 'value' | 'name'>;

export const ParameterReadOnlyValue = ({ value, name }: ParameterReadOnlyValueProps) => {
    if (isBoolean(value)) {
        return <span aria-label={name}>{value ? 'On' : 'Off'}</span>;
    }

    if (Array.isArray(value) && value.length === 2) {
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

export const NumberEnumParameterField = ({
    parameter,
    onChange,
    isDisabled,
}: {
    parameter: NumberEnumConfigurableParameter;
    onChange: (parameter: NumberEnumConfigurableParameter) => void;
    isDisabled?: boolean;
}) => {
    const handleChange = (value: NumberEnumConfigurableParameter['value']) => {
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
            onSelectionChange={(key) => handleChange(key as NumberEnumConfigurableParameter['value'])}
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

const ParameterField = ({ parameter, onChange, isDisabled }: ParameterFieldProps) => {
    if (parameter.value_type === 'float' || parameter.value_type === 'int') {
        if (parameter.allowed_values === null) {
            const handleChange = (value: number) => {
                onChange({
                    ...parameter,
                    value,
                });
            };

            return (
                <NumberParameterField
                    // TODO: Remove assertion after API update
                    value={Number(parameter.value)}
                    minValue={parameter.min_value ?? null}
                    maxValue={parameter.max_value ?? null}
                    onChange={handleChange}
                    type={parameter.value_type}
                    isDisabled={isDisabled}
                    name={parameter.name}
                />
            );
        }
    }

    if (parameter.value_type === 'float_range') {
        const handleChange = (value: [number, number]) => {
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

    if (parameter.value_type === 'bool') {
        const handleChange = (value: boolean) => {
            onChange({
                ...parameter,
                value,
            });
        };

        return (
            <BooleanParameterField
                // TODO: Remove assertion after API update
                value={Boolean(parameter.value)}
                header={parameter.name}
                onChange={handleChange}
                isDisabled={isDisabled}
            />
        );
    }

    return null;
};

export const Parameter = ({ parameter, onChange, isDisabled, marginStart, isReadOnly }: ParameterProps) => {
    if (isReadOnly) {
        return <ParameterReadOnly parameter={parameter} marginStart={marginStart} />;
    }

    const handleReset = () => {
        onChange({ ...parameter, value: parameter.default_value } as ConfigurableParameter);
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
    parameters: ConfigurableParameter[];
    onChange: (parameter: ConfigurableParameter) => void;
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
