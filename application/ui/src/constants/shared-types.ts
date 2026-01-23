// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { components } from '../api/openapi-spec';

export type Label = components['schemas']['LabelView'];
export type DatasetItem = components['schemas']['DatasetItemView'];

// TODO: Update these types once the backend spec makes 'id' required
export type Model = components['schemas']['ModelView'] & { id: string };
export type ExtendedModel = components['schemas']['ExtendedModelView'] & { id: string };
export type ModelArchitecture = components['schemas']['ModelArchitectureView'];

export type ModelArchitectureWithPerformanceCategory = ModelArchitecture & { performanceCategory?: string };

export type ModelFormat = components['schemas']['ModelFormat'];
export type DatasetSubset = components['schemas']['DatasetItemSubset'];
export type Job = components['schemas']['JobView'];

export type MediaItemState = 'accepted' | 'rejected';
export type MediaStateMap = Map<string, MediaItemState>;

export type DeviceType = components['schemas']['DeviceType'];
export type RecommendedModelArchitectures = components['schemas']['TopPicks'];

export type TrainingDevice = {
    type: DeviceType;
    name: string;
};

export type DatasetRevision = {
    id: string;
    name: string;
};

export type Project = components['schemas']['ProjectView'];

export type TaskType = 'detection' | 'instance_segmentation' | 'classification';
