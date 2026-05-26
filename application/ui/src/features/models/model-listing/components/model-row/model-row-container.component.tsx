// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Model, ModelArchitectureWithPerformanceCategory } from '../../../../../constants/shared-types';
import { useGetModel } from '../../../hooks/api/use-get-model.hook';
import { useModelListing } from '../../provider/model-listing-provider';
import { ModelActions } from '../model-actions/model-actions.component';
import { ModelRow } from './model-row.component';

type ModelRowContainerProps = {
    model: Model;
    modelArchitecture: ModelArchitectureWithPerformanceCategory | undefined;
};

export const ModelRowContainer = ({ model, modelArchitecture }: ModelRowContainerProps) => {
    const { onExpandModel, groupBy, datasetRevisions } = useModelListing();
    const { data: parentRevisionModel } = useGetModel(model.parent_revision);
    const datasetRevision = datasetRevisions.find(({ id }) => id === model.training_info.dataset_revision_id);

    return (
        <>
            <ModelRow
                model={model}
                parentRevisionModel={parentRevisionModel}
                onExpandModel={onExpandModel}
                groupBy={groupBy}
                datasetRevision={datasetRevision}
                modelArchitecture={modelArchitecture}
            />
            <ModelActions model={model} />
        </>
    );
};
