// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useModelListing } from '../../provider/model-listing-provider';
import type { ArchitectureGroup, DatasetGroup } from '../../types';
import { ArchitectureGroupHeader } from './architecture-group-header.component';
import { DatasetGroupHeader } from './dataset-group-header.component';

type GroupHeaderProps = {
    data: DatasetGroup | ArchitectureGroup;
};

export const GroupHeader = ({ data }: GroupHeaderProps) => {
    const { groupBy } = useModelListing();

    if (groupBy === 'dataset') {
        return <DatasetGroupHeader dataset={data as DatasetGroup} />;
    }

    return <ArchitectureGroupHeader architecture={data as ArchitectureGroup} />;
};
