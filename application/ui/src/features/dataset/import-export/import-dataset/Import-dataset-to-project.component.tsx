// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, Dialog, DialogContainer, Divider, Heading, View } from '@geti/ui';

import { useImportDatasetDialogState } from '../../providers/export-import-dataset-dialog-provider.component';
import { ImportDatasetButtons } from './import-dataset-buttons/import-dataset-buttons.component';
import { ImportDropZone } from './import-drop-zone/import-drop-zone.component';
import { ImportProcess } from './import-process/import-process.component';
import { LabelMapping } from './label-mapping/label-mapping.component';

export const ImportDatasetToProject = () => {
    const { datasetImportDialogState, currentStep, currentStagedId } = useImportDatasetDialogState();

    return (
        <DialogContainer onDismiss={datasetImportDialogState.close}>
            {datasetImportDialogState.isOpen && (
                <Dialog aria-label={'import-dataset-dialog'} width={800}>
                    <Heading>Import dataset</Heading>
                    <Divider />
                    <Content minHeight={'size-5000'}>
                        <View height={'100%'} backgroundColor={'gray-50'}>
                            {currentStep === 'dropzone' && <ImportDropZone />}
                            {currentStep === 'preparing' && <ImportProcess />}
                            {currentStep === 'labelMapping' && (
                                <LabelMapping stagedDatasetId={String(currentStagedId)} />
                            )}
                        </View>
                    </Content>

                    <ImportDatasetButtons
                        currentStep={currentStep}
                        stagedDatasetId={String(currentStagedId)}
                        onClose={datasetImportDialogState.close}
                    />
                </Dialog>
            )}
        </DialogContainer>
    );
};
