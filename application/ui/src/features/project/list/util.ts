// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import dayjs from 'dayjs';

import { TaskType } from '../../../constants/shared-types';

export const formatCreationDate = (creationDate: string) => {
    return dayjs(creationDate).format('D MMMM YYYY | h:mm A');
};

export const MAP_PROJECT_TYPE_TO_TITLE: Record<TaskType, string> = {
    detection: 'Object detection',
    classification: 'Classification',
    instance_segmentation: 'Instance segmentation',
};
