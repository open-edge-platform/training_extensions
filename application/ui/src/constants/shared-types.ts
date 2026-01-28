// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { components } from '../api/openapi-spec';

export type Label = components['schemas']['LabelView'];

export type Model = components['schemas']['ModelView'];
export type ExtendedModel = components['schemas']['ExtendedModelView'];
export type ModelArchitecture = components['schemas']['ModelArchitectureView'];
export type ModelArchitectureWithPerformanceCategory = ModelArchitecture & { performanceCategory?: string };
export type ModelFormat = components['schemas']['ModelFormat'];
export type RecommendedModelArchitectures = components['schemas']['TopPicks'];

export type Job = components['schemas']['JobView'];

export type Media = components['schemas']['MediaView'];
export type MediaItemState = 'accepted' | 'rejected';
export type MediaStateMap = Map<string, MediaItemState>;

export type DeviceType = components['schemas']['DeviceType'];
export type TrainingDevice = {
    type: DeviceType;
    name: string;
};

export type DatasetSubset = components['schemas']['DatasetItemSubset'];
export type DatasetItem = components['schemas']['DatasetItemView'];
export type DatasetRevision = {
    id: string;
    name: string;
};

export type Project = components['schemas']['ProjectView'];

export type TaskType = 'detection' | 'instance_segmentation' | 'classification';

export type ImagesFolderSourceConfig = components['schemas']['ImagesFolderSourceConfigView'];
export type IPCameraSourceConfig = components['schemas']['IPCameraSourceConfigView'];
export type USBCameraSourceConfig = components['schemas']['USBCameraSourceConfigView'];
export type VideoFileSourceConfig = components['schemas']['VideoFileSourceConfigView'];
export type DisconnectedSourceConfig = components['schemas']['DisconnectedSourceConfigView'];

export type SourceConfig =
    | DisconnectedSourceConfig
    | USBCameraSourceConfig
    | IPCameraSourceConfig
    | VideoFileSourceConfig
    | ImagesFolderSourceConfig;

export type SourceConfigPayload = Exclude<SourceConfig, DisconnectedSourceConfig>;

export type AnnotationDTO = components['schemas']['DatasetItemAnnotation-Input'];
