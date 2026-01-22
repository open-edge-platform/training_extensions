// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

interface ParameterBase {
    key: string;
    name: string;
    description: string;
}

export interface NumberParameter extends ParameterBase {
    type: 'int' | 'float';
    value: number;
    min_value: number | null;
    max_value: number | null;
    default_value: number;
}

export interface ArrayParameter extends ParameterBase {
    type: 'array';
    value: number[];
    default_value: number[];
}

export interface BoolParameter extends ParameterBase {
    type: 'bool';
    value: boolean;
    default_value: boolean;
}

interface EnumParameter<T extends boolean | number> extends ParameterBase {
    type: 'enum';
    value: T;
    default_value: T;
    allowed_values: T[];
}

export interface StaticParameter extends ParameterBase {
    value: number | boolean;
}

export type EnumConfigurationParameter = EnumParameter<number>;

export type ConfigurationParameter = BoolParameter | NumberParameter | EnumConfigurationParameter | ArrayParameter;

export type DatasetPreparationParameters = {
    subset_split: ConfigurationParameter[];
    filtering: Record<string, ConfigurationParameter[]>;
    augmentation: Record<string, ConfigurationParameter[]>;
};

export type TrainingParameters = (ConfigurationParameter | Record<string, ConfigurationParameter[]>)[];

export interface TrainingConfiguration {
    dataset_preparation: DatasetPreparationParameters;
    training: TrainingParameters;
    evaluation: ConfigurationParameter[];
}

export interface TrainedModelConfiguration extends Omit<TrainingConfiguration, 'dataset_preparation'> {
    dataset_preparation: Pick<DatasetPreparationParameters, 'augmentation'>;
    advanced_configuration: StaticParameter[];
}

export type TrainingConfigurationUpdatePayload = TrainingConfiguration;
