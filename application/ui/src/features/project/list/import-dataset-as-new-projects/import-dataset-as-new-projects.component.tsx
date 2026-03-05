// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Button, ButtonGroup, Content, Dialog, DialogContainer, Divider, Heading, View } from '@geti/ui';
import { OverlayTriggerState } from '@react-stately/overlays';

import {
    FileUploadedResponse,
    ImportUploadFile,
} from '../../../../components/import-upload-file/import-upload-file.component';
import { useImportDatasetAsNewProject } from '../../../../hooks/localStorage/use-import-dataset-as-new-project.hook';
import { ImportDatasetAsNewProjectState } from '../../../dataset/import-export/import-dataset/util';
import { ImportProcess } from './import-process/import-process.component';
import { ProgressStepper } from './ProgressStepper/progress-stepper.component';

type ImportDatasetAsNewProjectProps = {
    dialogState: OverlayTriggerState;
};

export const ImportDatasetAsNewProject = ({ dialogState }: ImportDatasetAsNewProjectProps) => {
    const { appendImportEntry } = useImportDatasetAsNewProject();
    const [currentStep, setCurrentStep] = useState<ImportDatasetAsNewProjectState>('uploading');
    const [stagedDatasetId, setStagedDatasetId] = useState<string | null>(null);

    const handleFileUploaded = (response: FileUploadedResponse) => {
        appendImportEntry({ ...response, step: 'preparing', importJobId: null });
        setCurrentStep('preparing');
        setStagedDatasetId(response.stagedDatasetId);
    };

    return (
        <DialogContainer onDismiss={dialogState.close}>
            {dialogState.isOpen && (
                <Dialog aria-label={'import-dataset-dialog'} width={800}>
                    <Heading>Create project from a dataset - Import</Heading>
                    <Divider />
                    <Content minHeight={'size-5000'}>
                        <ProgressStepper />

                        <View backgroundColor={'gray-50'}>
                            {currentStep === 'uploading' && <ImportUploadFile onFileUploaded={handleFileUploaded} />}
                            {currentStep === 'preparing' && (
                                <ImportProcess
                                    stagedDatasetId={String(stagedDatasetId)}
                                    setCurrentStep={setCurrentStep}
                                />
                            )}
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
