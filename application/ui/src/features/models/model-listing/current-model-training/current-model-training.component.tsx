// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Heading, View } from '@geti/ui';
import { useCancelJob, useGetCurrentTrainingJob } from 'hooks/api/jobs.hook';

import { type DatasetRevision } from '../../../../constants/shared-types';
import { useGetTaskModelArchitectures } from '../../hooks/api/use-get-model-architectures.hook';
import { useStreamJobLogs } from '../../training-logs/hooks/use-stream-job-logs.hook';
import { ModelsTableHeader } from '../components/models-table-header.component';
import { GroupByMode } from '../types';
import { TrainingModelRow } from './training-model-row.component';

type CurrentModelTrainingProps = {
    groupBy: GroupByMode;
    datasetRevisions: DatasetRevision[];
};

export const CurrentModelTraining = ({ groupBy, datasetRevisions }: CurrentModelTrainingProps) => {
    const activeTrainingJob = useGetCurrentTrainingJob();
    const cancelJobMutation = useCancelJob();

    const { modelArchitectures } = useGetTaskModelArchitectures();

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
        >
            <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                Current training
            </Heading>

            <View backgroundColor={'gray-75'}>
                <ModelsTableHeader />

                <TrainingModelRow
                    job={activeTrainingJob}
                    onCancel={handleCancelTraining}
                    groupBy={groupBy}
                    datasetRevisions={datasetRevisions}
                    modelArchitectures={modelArchitectures}
                />
            </View>
        </Flex>
    );
};
