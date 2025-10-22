// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Grid, minmax, RadioGroup, repeat } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { $api } from 'src/api/client';

import { ModelType } from './model-type.component';

interface ModelTypesListProps {
    selectedModelArchitectureId: string | null;
    setSelectedModelArchitectureId: (modelTemplateId: string | null) => void;
}
export const ModelTypesList = ({
    selectedModelArchitectureId,
    setSelectedModelArchitectureId,
}: ModelTypesListProps) => {
    const project_id = useProjectIdentifier();

    const { data: projectData } = $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id } },
    });

    const { data: modelArchitecturesResponse } = $api.useSuspenseQuery('get', '/api/model_architectures', {
        params: { query: { task: projectData.task.task_type } },
    });

    return (
        <RadioGroup width={'100%'} aria-label='model type' onChange={setSelectedModelArchitectureId}>
            <Grid columns={repeat('auto-fit', minmax('size-3400', '1fr'))} gap={'size-250'}>
                {modelArchitecturesResponse.model_architectures.map((architecture) => (
                    <ModelType
                        key={architecture.id}
                        modelArchitecture={architecture}
                        selectedModelArchitectureId={selectedModelArchitectureId}
                    />
                ))}
            </Grid>
        </RadioGroup>
    );
};
