// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export type Project = {
    id: string;
    name: string;
    task: {
        task_type: 'segmentation' | 'detection' | 'classification';
        exclusive_labels: boolean;
        labels: Array<{
            id: string;
            name: string;
            color: string | null;
            hotkey: string | null;
        }>;
    };
};
