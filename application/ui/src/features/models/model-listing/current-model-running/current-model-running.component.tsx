// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Flex, Heading, View } from '@geti/ui';
import { useCancelJob, useGetCurrentRunningJobs } from 'hooks/api/jobs/jobs.hook';
import { isEmpty, isNil } from 'lodash-es';

import { type DatasetRevision } from '../../../../constants/shared-types';
import { useGetTaskModelArchitectures } from '../../hooks/api/use-get-model-architectures.hook';
import { ModelsTableHeader } from '../components/models-table-header.component';
import { GroupByMode } from '../types';
import { RunningModelRow } from './running-model-row.component';

type CurrentModelRunningProps = {
    groupBy: GroupByMode;
    datasetRevisions: DatasetRevision[];
};

export const CurrentModelRunning = ({ groupBy, datasetRevisions }: CurrentModelRunningProps) => {
    const cancelJobMutation = useCancelJob();
    const activeRunningJobs = useGetCurrentRunningJobs();
    const { modelArchitectures } = useGetTaskModelArchitectures();

    const handleCancelRunning = (jobId: string | undefined) => {
        if (jobId) {
            cancelJobMutation.mutate({ params: { path: { job_id: jobId } } });
        }
    };

    if (isNil(activeRunningJobs) || isEmpty(activeRunningJobs)) {
        return null;
    }

    return (
        <Flex
            width={'100%'}
            gap={'size-200'}
            direction={'column'}
            UNSAFE_style={{ padding: 'var(--spectrum-global-dimension-size-300)' }}
        >
            <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                Currently running
            </Heading>

            <View backgroundColor={'gray-75'}>
                <ModelsTableHeader />

                <View>
                    {activeRunningJobs.map((job) => (
                        <RunningModelRow
                            key={job.job_id}
                            job={job}
                            onCancel={() => handleCancelRunning(job.job_id)}
                            groupBy={groupBy}
                            datasetRevisions={datasetRevisions}
                            modelArchitectures={modelArchitectures}
                        />
                    ))}
                </View>
            </View>
        </Flex>
    );
};
