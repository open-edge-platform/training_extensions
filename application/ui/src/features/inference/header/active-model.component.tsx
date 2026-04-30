// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { Item, Key, Picker } from '@geti/ui';
import { usePatchPipeline } from 'hooks/api/pipeline.hook';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { isEmpty } from 'lodash-es';

import { useGetActiveModel } from '../../models/hooks/api/use-get-active-model.hook';
import { useGetSuccessfulModels } from '../../models/hooks/api/use-get-models.hook';
import { getAllModelsWithOpenVinoVariants, getModelIdentifierPayload } from '../../models/utils';

export const ActiveModel = () => {
    const { data: models } = useGetSuccessfulModels();
    const activeModel = useGetActiveModel();
    const projectId = useProjectIdentifier();
    const updatePipeline = usePatchPipeline();

    const allModelsWithOpenVinoQuantizedModels = useMemo(() => getAllModelsWithOpenVinoVariants(models), [models]);

    const handleChange = (key: Key | null) => {
        if (key === null) {
            return;
        }

        const selectedModel = allModelsWithOpenVinoQuantizedModels.find((model) => model.id === key);

        if (selectedModel === undefined) {
            return;
        }

        const body = getModelIdentifierPayload(selectedModel);

        updatePipeline.mutate({
            params: { path: { project_id: projectId } },
            body,
        });
    };

    if (isEmpty(allModelsWithOpenVinoQuantizedModels)) {
        return null;
    }

    return (
        <>
            <Picker
                aria-label={'active model'}
                label={'Model'}
                labelPosition={'side'}
                items={allModelsWithOpenVinoQuantizedModels}
                onSelectionChange={handleChange}
                selectedKey={activeModel?.id ?? null}
                minWidth={'size-3400'}
            >
                {(item) => <Item>{item.name}</Item>}
            </Picker>
        </>
    );
};
