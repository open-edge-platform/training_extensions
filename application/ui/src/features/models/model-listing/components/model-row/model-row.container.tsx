// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Model } from '../../../../../constants/shared-types';
import { useGetModel } from '../../../hooks/api/use-get-model.hook';
import { useModelListing } from '../../provider/model-listing-provider';
import { ModelActions } from '../model-actions/model-actions.component';
import { ModelRow } from './model-row.component';

type ModelRowContainerProps = {
    model: Model;
};

export const ModelRowContainer = ({ model }: ModelRowContainerProps) => {
    const { activeModelArchitectureId, onExpandModel } = useModelListing();
    const { data: parentRevisionModel } = useGetModel(model.parent_revision);

    return (
        <>
            <ModelRow
                model={model}
                activeModelArchitectureId={activeModelArchitectureId}
                parentRevisionModel={parentRevisionModel}
                onExpandModel={onExpandModel}
            />
            <ModelActions model={model} />
        </>
    );
};
