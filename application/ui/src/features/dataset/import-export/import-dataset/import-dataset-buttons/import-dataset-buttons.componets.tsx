// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';
import { useCancelJob } from 'hooks/api/jobs.hook';
import { usePrepareImportDataset } from 'hooks/localStorage/use-prepare-import-dataset.hook';

import { ImportDatasetState } from '../util';

type ImportDatasetButtonsProps = {
    onClose: () => void;
    currentState: ImportDatasetState;
};

export const ImportDatasetButtons = ({ currentState, onClose }: ImportDatasetButtonsProps) => {
    const cancelJobMutation = useCancelJob();
    const { getLsPreparingImportId } = usePrepareImportDataset();

    const handleCancelJob = (jobId: string) => {
        cancelJobMutation.mutate({ params: { path: { job_id: jobId } } }, { onSuccess: onClose });
    };

    if (currentState === 'preparing') {
        return (
            <ButtonGroup>
                <Button
                    variant='negative'
                    isPending={cancelJobMutation.isPending}
                    isDisabled={cancelJobMutation.isPending}
                    onPress={() => handleCancelJob(String(getLsPreparingImportId()?.id))}
                >
                    Cancel
                </Button>
                <Button
                    onPress={onClose}
                    variant='secondary'
                    isPending={cancelJobMutation.isPending}
                    isDisabled={cancelJobMutation.isPending}
                >
                    Hide
                </Button>
            </ButtonGroup>
        );
    }

    return (
        <ButtonGroup>
            <Button onPress={onClose} variant='secondary'>
                Cancel
            </Button>
        </ButtonGroup>
    );
};
