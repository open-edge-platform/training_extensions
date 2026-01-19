// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { useGetCurrentTrainingJob } from 'hooks/api/jobs.hook';

import { useGetModel } from '../../hooks/api/use-get-model.hook';
import { ModelRow } from '../components/model-row/model-row.component';
import { ModelsTableHeader } from '../components/models-table-header.component';

export const CurrentModelTraining = () => {
    const activeTrainingJob = useGetCurrentTrainingJob();
    const activeTrainingJobModel = useGetModel(activeTrainingJob?.metadata.model.id);

    if (!activeTrainingJob || !activeTrainingJobModel?.data) {
        return null;
    }

    const model = activeTrainingJobModel.data;

    return (
        <Flex>
            <ModelsTableHeader />
            <ModelRow model={model} />
        </Flex>
    );
};
