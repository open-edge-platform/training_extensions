// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';
import { useImportDatasetAsNewProject } from 'hooks/localStorage/use-import-dataset-as-new-project.hook';

import { ImportProcessButtons } from '../../../../../components/import-job-process/import-process-buttons.component';
import { ImportDatasetAsNewProjectState } from '../../../../dataset/import-export/import-dataset/util';

type ImportDatasetButtonsProps = {
    onClose: () => void;
    stagedDatasetId: string | null;
    currentStep: ImportDatasetAsNewProjectState;
};

export const ImportDatasetButtons = ({ currentStep, stagedDatasetId, onClose }: ImportDatasetButtonsProps) => {
    const { getImportEntry } = useImportDatasetAsNewProject();
    const { prepareJobId } = getImportEntry(stagedDatasetId ?? '') ?? { prepareJobId: null };

    if (stagedDatasetId === null || prepareJobId === null) {
        return (
            <ButtonGroup>
                <Button onPress={onClose} variant='secondary'>
                    Cancel
                </Button>
            </ButtonGroup>
        );
    }

    if (currentStep === 'preparing') {
        return <ImportProcessButtons onClose={onClose} prepareJobId={prepareJobId} stagedDatasetId={stagedDatasetId} />;
    }

    return (
        <ButtonGroup>
            <Button onPress={onClose} variant='secondary'>
                Cancel
            </Button>
        </ButtonGroup>
    );
};
