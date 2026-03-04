// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';

import { useImportDatasetToProject } from '../../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import { ImportProcessButtons } from '../import-process/import-process-buttons.component';
import { LabelMappingButtons } from '../label-mapping/label-mapping-buttons.component';
import { ImportDatasetToProjectState } from '../util';

type ImportDatasetButtonsProps = {
    onClose: () => void;
    stagedDatasetId: string | null;
    currentStep: ImportDatasetToProjectState;
};

export const ImportDatasetButtons = ({ currentStep, stagedDatasetId, onClose }: ImportDatasetButtonsProps) => {
    const { getImportEntry } = useImportDatasetToProject();
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

    if (currentStep === 'labelMapping') {
        return <LabelMappingButtons stagedDatasetId={stagedDatasetId} onClose={onClose} />;
    }

    return (
        <ButtonGroup>
            <Button onPress={onClose} variant='secondary'>
                Cancel
            </Button>
        </ButtonGroup>
    );
};
