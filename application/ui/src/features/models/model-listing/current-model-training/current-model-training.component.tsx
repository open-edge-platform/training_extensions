// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Heading, View } from '@geti/ui';
import { useCancelJob, useGetCurrentTrainingJob } from 'hooks/api/jobs.hook';

import { ModelsTableHeader } from '../components/models-table-header.component';
import { TrainingModelRow } from './training-model-row.component';

export const CurrentModelTraining = () => {
    const activeTrainingJob = useGetCurrentTrainingJob();
    const cancelJobMutation = useCancelJob();

    const handleCancelTraining = () => {
        if (activeTrainingJob?.job_id) {
            cancelJobMutation.mutate({ params: { path: { job_id: activeTrainingJob.job_id } } });
        }
    };

    if (!activeTrainingJob) {
        return null;
    }

    return (
        <Flex
            gap={'size-200'}
            direction={'column'}
            UNSAFE_style={{ padding: 'var(--spectrum-global-dimension-size-300)' }}
            marginBottom={'size-200'}
        >
            <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                Current training
            </Heading>

            <View backgroundColor={'gray-75'}>
                <ModelsTableHeader />

                <TrainingModelRow job={activeTrainingJob} onCancel={handleCancelTraining} />
            </View>
        </Flex>
    );
};
