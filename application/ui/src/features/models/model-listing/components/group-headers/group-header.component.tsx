// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Model, ModelArchitectureWithPerformanceCategory } from '../../../../../constants/shared-types';
import { useModelListing } from '../../provider/model-listing-provider';
import type { ArchitectureGroup, DatasetGroup } from '../../types';
import { ArchitectureGroupHeader } from './architecture-group-header.component';
import { DatasetGroupHeader } from './dataset-group-header.component';

type GroupHeaderProps = {
    data: DatasetGroup | ArchitectureGroup;
    models: Model[];
    modelArchitectures: ModelArchitectureWithPerformanceCategory[];
};

const getLatestModelRevisionId = (models: Model[]): string | undefined => {
    const latestModelRevisionId = models.length
        ? models.reduce((latest, model) => {
              const latestEndTime = latest.training_info?.end_time ?? '';
              const currentEndTime = model.training_info?.end_time ?? '';

              return currentEndTime > latestEndTime ? model : latest;
          }).id
        : undefined;

    return latestModelRevisionId;
};

export const GroupHeader = ({ data, models, modelArchitectures }: GroupHeaderProps) => {
    const { groupBy } = useModelListing();

    if (groupBy === 'dataset') {
        return <DatasetGroupHeader dataset={data as DatasetGroup} />;
    }

    const architecture = data as ArchitectureGroup;
    const modelArchitecture = modelArchitectures.find(({ id }) => id === architecture.id);
    const latestModelRevisionId = getLatestModelRevisionId(models);

    return (
        <ArchitectureGroupHeader architecture={modelArchitecture} preSelectedModelRevisionId={latestModelRevisionId} />
    );
};
