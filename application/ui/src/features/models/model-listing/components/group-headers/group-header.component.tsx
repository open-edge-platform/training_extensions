// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ArchitectureGroup, DatasetGroup, GroupByMode } from '../../types';
import { ArchitectureGroupHeader } from './architecture-group-header.component';
import { DatasetGroupHeader } from './dataset-group-header.component';

type GroupHeaderProps = {
    groupBy: GroupByMode;
    data: DatasetGroup | ArchitectureGroup;
};

export const GroupHeader = ({ groupBy, data }: GroupHeaderProps) => {
    if (groupBy === 'dataset') {
        return <DatasetGroupHeader dataset={data as DatasetGroup} />;
    }

    return <ArchitectureGroupHeader architecture={data as ArchitectureGroup} />;
};
