// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Content, Dialog, DialogContainer, Divider, Heading, View } from '@geti/ui';
import { OverlayTriggerState } from '@react-stately/overlays';

import { useLabelMappingImportDataset } from '../../../../hooks/localStorage/use-label-mapping-import-dataset.hook';
import { usePrepareImportDataset } from '../../../../hooks/localStorage/use-prepare-import-dataset.hook';
import { ImportDatasetButtons } from './import-dataset-buttons/import-dataset-buttons.component';
import { ImportDropZone } from './import-drop-zone/import-drop-zone.component';
import { ImportProcess } from './import-process/import-process.component';
import { LabelMapping } from './label-mapping/label-mapping.component';
import { ImportDatasetState } from './util';

type ImportDatasetProps = {
    dialogState: OverlayTriggerState;
};

export const ImportDataset = ({ dialogState }: ImportDatasetProps) => {
    const { getLsPreparingImport } = usePrepareImportDataset();
    const { getLsLabelMappingImport } = useLabelMappingImportDataset();

    const [currentState, setCurrentState] = useState<ImportDatasetState>(() => {
        if (getLsLabelMappingImport() !== null) {
            return 'labelMapping';
        }

        if (getLsPreparingImport() !== null) {
            return 'preparing';
        }

        return 'dropzone';
    });

    const handleNextStep = (step: ImportDatasetState) => {
        setCurrentState(step);
    };

    return (
        <DialogContainer onDismiss={dialogState.close}>
            {dialogState.isOpen && (
                <Dialog aria-label={'import-dataset-dialog'} width={800}>
                    <Heading>Import dataset</Heading>
                    <Divider />
                    <Content>
                        <View backgroundColor={'gray-50'}>
                            {currentState === 'dropzone' && <ImportDropZone onNextStep={handleNextStep} />}
                            {currentState === 'preparing' && <ImportProcess onNextStep={handleNextStep} />}
                            {currentState === 'labelMapping' && <LabelMapping onNextStep={handleNextStep} />}
                        </View>
                    </Content>

                    <ImportDatasetButtons currentState={currentState} onClose={dialogState.close} />
                </Dialog>
            )}
        </DialogContainer>
    );
};
