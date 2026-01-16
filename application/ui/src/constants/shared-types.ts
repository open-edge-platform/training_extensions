// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { components } from '../api/openapi-spec';

export type Label = components['schemas']['LabelView'];
export type DatasetItem = components['schemas']['DatasetItemView'];
export type Model = components['schemas']['ModelView'];
export type ModelArchitecture = components['schemas']['ModelArchitectureView'];
export type DatasetSubset = components['schemas']['DatasetItemSubset'];

export type MediaItemState = 'accepted' | 'rejected';
export type MediaStateMap = Map<string, MediaItemState>;

export type DeviceType = components['schemas']['DeviceType'];

export type TrainingDevices = {
    type: DeviceType;
    name: string;
};

export type DatasetRevision = {
    id: string;
    name: string;
};
