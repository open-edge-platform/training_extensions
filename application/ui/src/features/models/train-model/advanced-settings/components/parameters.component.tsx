// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key, ReactNode } from 'react';

import {
    Content,
    ContextualHelp,
    DimensionValue,
    Flex,
    Grid,
    Item,
    minmax,
    Picker,
    Text,
    ToggleButtons,
    View,
} from '@geti/ui';
import { isBoolean, isFunction } from 'lodash-es';

import {
    BoolConfigurableParameter,
    ConfigurableParameter,
    ConfigurableParameterGroup,
    NumberEnumConfigurableParameter,
    StringEnumConfigurableParameter,
    TrainingConfigurationParameter,
} from '../../../../../constants/shared-types';
import { isParameter, isParameterGroup } from '../../../model-listing/model-training-parameters/utils';
import { isBoolEnableParameter, isEnumNumberParameter, isEnumStringParameter, isNumberParameter } from '../utils';
import { BooleanParameterField } from './boolean-parameter-field.component';
import { NumberParameterField } from './number-parameter-field.component';
import { RangeParameterField } from './range-parameter-field/range-parameter-field.component';
import { ResetButton } from './reset-button.component';

type ParameterGroupWithParameters = Omit<ConfigurableParameterGroup, 'parameters'> & {
    parameters: ConfigurableParameter[];
};

type ParametersProps = {
    parameters: TrainingConfigurationParameter[];
    onChange: (parameter: ConfigurableParameter, groupKey?: string) => void;
    isReadOnly?: boolean;
    isDisabled?: boolean;
    marginStart?: DimensionValue;
};

type ParametersGroupProps = {
    parametersGroup: ConfigurableParameterGroup;
    onChange: (parameter: ConfigurableParameter, groupKey: string) => void;
    isReadOnly?: boolean;
    isDisabled?: boolean;
    marginStart?: DimensionValue;
};

type ParametersEnableGroupParameters = ConfigurableParameterGroup & {
    parameters: [BoolConfigurableParameter, ...ConfigurableParameterGroup[]];
};

type ParametersEnableGroupProps = {
    parameters: ParametersEnableGroupParameters;
    onChange: (parameter: ConfigurableParameter, groupKey: string) => void;
    isReadOnly: boolean;
    isDisabled?: boolean;
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
    header: string;
    description: string;
    parameter: ConfigurableParameter;
    onChange: (parameter: ConfigurableParameter) => void;
    isDisabled?: boolean;
    marginStart?: DimensionValue;
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
    marginStart?: DimensionValue;
};

type ParameterNameProps = {
    name: string;
    description: string;
    gridColumn?: string;
    marginStart?: DimensionValue;
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
    marginStart?: DimensionValue;
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

const StringEnumParameterField = ({
    parameter,
    onChange,
    isDisabled,
}: {
    parameter: StringEnumConfigurableParameter;
    onChange: (parameter: StringEnumConfigurableParameter) => void;
    isDisabled?: boolean;
}) => {
    const handleChange = (value: Key) => {
        onChange({
            ...parameter,
            value: value.toString(),
        });
    };

    const items = parameter.allowed_values.map((value) => ({ value }));

    return (
        <Picker
            isDisabled={isDisabled}
            items={items}
            selectedKey={parameter.value.toString()}
            onSelectionChange={(key) => key !== null && handleChange(key)}
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

export const NumberEnumParameterField = ({
    parameter,
    onChange,
    isDisabled,
}: {
    parameter: NumberEnumConfigurableParameter;
    onChange: (parameter: NumberEnumConfigurableParameter) => void;
    isDisabled?: boolean;
}) => {
    const handleChange = (value: Key) => {
        onChange({
            ...parameter,
            value: Number(value),
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
            onSelectionChange={(key) => key !== null && handleChange(key)}
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
    if (isEnumStringParameter(parameter)) {
        return <StringEnumParameterField parameter={parameter} onChange={onChange} isDisabled={isDisabled} />;
    }

    if (isEnumNumberParameter(parameter)) {
        return <NumberEnumParameterField parameter={parameter} onChange={onChange} />;
    }

    if (isNumberParameter(parameter)) {
        const handleChange = (value: number) => {
            onChange({
                ...parameter,
                value,
            });
        };

        return (
            <NumberParameterField
                value={parameter.value}
                minValue={parameter.min_value ?? null}
                maxValue={parameter.max_value ?? null}
                onChange={handleChange}
                type={parameter.value_type}
                isDisabled={isDisabled}
                name={parameter.name}
            />
        );
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
                value={parameter.value}
                header={parameter.name}
                onChange={handleChange}
                isDisabled={isDisabled}
            />
        );
    }

    return null;
};

export const Parameter = ({
    header,
    description,
    parameter,
    onChange,
    isDisabled,
    marginStart,
    isReadOnly,
}: ParameterProps) => {
    if (isReadOnly) {
        return <ParameterReadOnly parameter={parameter} marginStart={marginStart} />;
    }

    const handleReset = () => {
        onChange({ ...parameter, value: parameter.default_value } as ConfigurableParameter);
    };

    return (
        <ParameterLayout header={header} description={description} onReset={handleReset} marginStart={marginStart}>
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
    return parameters.map((parameter) => (
        <Parameter
            key={parameter.name}
            header={parameter.name}
            description={parameter.description}
            parameter={parameter}
            onChange={onChange}
            isReadOnly={isReadOnly}
        />
    ));
};

const ParametersContainer = ({
    children,
    isReadOnly,
    columnGap = 'size-300',
    rowGap = 'size-300',
}: {
    children: ReactNode;
    isReadOnly?: boolean;
    columnGap?: DimensionValue;
    rowGap?: DimensionValue;
}) => {
    const columns = isReadOnly ? ['size-3000', '1fr'] : ['size-3000', minmax('size-3400', '1fr'), 'size-400'];

    return (
        <Grid columns={columns} columnGap={columnGap} rowGap={rowGap} alignItems={'center'}>
            {children}
        </Grid>
    );
};

const isBoolEnableGroup = (parameter: TrainingConfigurationParameter): parameter is ParametersEnableGroupParameters => {
    return (
        isParameterGroup(parameter) &&
        isParameter(parameter.parameters[0]) &&
        isBoolEnableParameter(parameter.parameters[0])
    );
};

export const ParametersEnableGroup = ({ parameters, onChange, isReadOnly, isDisabled }: ParametersEnableGroupProps) => {
    const handleChange = (groupKey: string) => (parameter: ConfigurableParameter) => {
        onChange(parameter, groupKey);
    };

    const [enableParameter, ...configurableParameters] = parameters.parameters;

    return (
        <ParametersContainer
            key={parameters.key}
            rowGap={configurableParameters.length > 0 ? 'size-150' : 'size-0'}
            isReadOnly={isReadOnly}
        >
            <Parameter
                header={parameters.name}
                description={parameters.description}
                parameter={enableParameter}
                onChange={handleChange(parameters.key)}
                isReadOnly={isReadOnly}
                isDisabled={isDisabled}
            />

            <Flex direction={'column'} gap={'size-150'}>
                <Parameters
                    parameters={configurableParameters}
                    onChange={onChange}
                    isReadOnly={isReadOnly}
                    isDisabled={!enableParameter.value}
                    marginStart={'size-200'}
                />
            </Flex>
        </ParametersContainer>
    );
};

export const ParametersGroup = ({
    parametersGroup,
    onChange,
    isReadOnly = false,
    isDisabled,
    marginStart,
}: ParametersGroupProps) => {
    const handleChange = (groupKey: string) => (parameter: ConfigurableParameter) => {
        onChange(parameter, groupKey);
    };

    if (isBoolEnableGroup(parametersGroup)) {
        return (
            <ParametersEnableGroup
                parameters={parametersGroup}
                onChange={onChange}
                isReadOnly={isReadOnly}
                isDisabled={isDisabled}
            />
        );
    }

    return (
        <Flex direction={'column'} gap={'size-300'}>
            {parametersGroup.parameters.map((parameter) => {
                if (isParameter(parameter)) {
                    return (
                        <ParametersContainer key={parameter.name}>
                            <Parameter
                                header={parameter.name}
                                description={parameter.description}
                                parameter={parameter}
                                onChange={handleChange(parameter.key)}
                                isReadOnly={isReadOnly}
                                isDisabled={isDisabled}
                                marginStart={marginStart}
                            />
                        </ParametersContainer>
                    );
                }

                if (isBoolEnableGroup(parameter)) {
                    return (
                        <ParametersEnableGroup
                            key={parameter.key}
                            parameters={parameter}
                            onChange={onChange}
                            isReadOnly={isReadOnly}
                            isDisabled={isDisabled}
                        />
                    );
                }

                return (
                    <Parameters
                        key={parameter.key}
                        parameters={parameter.parameters}
                        onChange={onChange}
                        isReadOnly={isReadOnly}
                        isDisabled={isDisabled}
                        marginStart={marginStart}
                    />
                );
            })}
        </Flex>
    );
};

export const Parameters = ({
    parameters,
    onChange,
    isReadOnly = false,
    isDisabled = false,
    marginStart,
}: ParametersProps) => {
    return (
        <>
            {parameters.map((parameter) => {
                if (isParameter(parameter)) {
                    return (
                        <ParametersContainer key={parameter.name}>
                            <Parameter
                                header={parameter.name}
                                description={parameter.description}
                                parameter={parameter}
                                onChange={onChange}
                                isReadOnly={isReadOnly}
                                isDisabled={isDisabled}
                                marginStart={marginStart}
                            />
                        </ParametersContainer>
                    );
                }

                return (
                    <ParametersGroup
                        key={parameter.key}
                        parametersGroup={parameter}
                        onChange={onChange}
                        isReadOnly={isReadOnly}
                        isDisabled={isDisabled}
                        marginStart={marginStart}
                    />
                );
            })}
        </>
    );
};

Parameters.Container = ParametersContainer;
Parameter.Layout = ParameterLayout;
