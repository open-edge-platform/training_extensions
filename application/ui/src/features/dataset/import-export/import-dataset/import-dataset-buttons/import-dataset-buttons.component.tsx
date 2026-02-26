// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup } from '@geti/ui';

import { useImportDatasetToProject } from '../../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import { ImportProcessButtons } from '../import-process/import-process-buttons.component';
import { LabelMappingButtons } from '../label-mapping/label-mapping-buttons.component';
import { ImportDatasetState } from '../util';

type ImportDatasetButtonsProps = {
    onClose: () => void;
    currentState: ImportDatasetState;
};

export const ImportDatasetButtons = ({ currentState, onClose }: ImportDatasetButtonsProps) => {
    const { getLastImportEntry } = useImportDatasetToProject();
    const prepareJobId = getLastImportEntry()?.prepareJobId ?? null;

    if (currentState === 'preparing' && prepareJobId !== null) {
        return <ImportProcessButtons prepareJobId={prepareJobId} onClose={onClose} />;
    }

    if (currentState === 'labelMapping') {
        return <LabelMappingButtons onClose={onClose} />;
    }

    return (
        <ButtonGroup>
            <Button onPress={onClose} variant='secondary'>
                Cancel
            </Button>
        </ButtonGroup>
    );
};
