// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export type TaskType = 'detection' | 'instance_segmentation' | 'classification';

export type TaskOption = {
    id: string;
    imageSrc: string;
    title: string;
    description: string;
    verb: string;
    value: TaskType;
};
