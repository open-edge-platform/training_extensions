// Copyright (C) 2022-2025 Intel Corporation
// LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

import { useState } from 'react';

import { Grid, minmax, repeat } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { $api } from 'src/api/client';

import { ModelType } from './model-type.component';

export const ModelTypesList = () => {
    const projectId = useProjectIdentifier();
    const [selectedModelArchitectureId, setSelectedModelArchitectureId] = useState<string | null>(null);
    const { data: projectData } = $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });

    const { data: modelArchitecturesResponse } = $api.useSuspenseQuery('get', '/api/model_architectures', {
        params: { query: { task: projectData.task.task_type } },
    });

    return (
        <Grid columns={repeat('auto-fit', minmax('size-3400', '1fr'))} gap={'size-250'}>
            {modelArchitecturesResponse.model_architectures.map((architecture) => (
                <ModelType
                    key={architecture.id}
                    activeModelTemplateId={null}
                    modelArchitecture={architecture}
                    selectedModelArchitectureId={selectedModelArchitectureId}
                    onChangeSelectedTemplateId={setSelectedModelArchitectureId}
                />
            ))}
        </Grid>
    );
};
