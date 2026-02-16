// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Heading, View } from '@geti/ui';
import { OverlayTriggerState } from '@react-stately/overlays';

import { ImportDropZone } from './import-drop-zone/import-drop-zone.component';
import { ImportProcess } from './import-process/import-process.component';
import { ImportDatasetState } from './util';

type ImportDatasetProps = {
    dialogState: OverlayTriggerState;
};

export const ImportDataset = ({ dialogState }: ImportDatasetProps) => {
    const [currentState, setCurrentState] = useState<ImportDatasetState>('dropzone');

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
                            {currentState === 'process' && <ImportProcess />}
                        </View>
                    </Content>

                    <ButtonGroup>
                        <Button onPress={dialogState.close} variant='secondary'>
                            Cancel
                        </Button>
                    </ButtonGroup>
                </Dialog>
            )}
        </DialogContainer>
    );
};
