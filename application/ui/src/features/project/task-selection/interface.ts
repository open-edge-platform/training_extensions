// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { TaskType } from '../../../constants/shared-types';

export type TaskOption = {
    id: string;
    imageSrc: string;
    title: string;
    description: string;
    verb: string;
    value: TaskType;
};
