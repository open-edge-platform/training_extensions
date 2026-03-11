// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, Dialog, DialogContainer, Divider, Heading, View } from '@geti/ui';
import { OverlayTriggerState } from '@react-stately/overlays';

import {
    FileUploadedResponse,
    ImportUploadFile,
} from '../../../../components/import-upload-file/import-upload-file.component';
import { useImportDatasetAsNewProject } from '../../../../hooks/localStorage/use-import-dataset-as-new-project.hook';
import { isNonEmptyString } from '../../../../shared/util';
import { useImportDatasetDialog } from '../../providers/import-dataset-dialog-provider.component';
import { ImportDatasetButtons } from './import-dataset-buttons/import-dataset-buttons.component';
import { ImportProcess } from './import-process/import-process.component';
import { ImportTaskSelection } from './import-task-selection/import-task-selection.component';
import { ProgressStepper } from './progress-stepper/progress-stepper.component';

import classes from './import-dataset-as-new-project.module.scss';

type ImportDatasetAsNewProjectProps = {
    dialogState: OverlayTriggerState;
};

export const ImportDatasetAsNewProject = ({ dialogState }: ImportDatasetAsNewProjectProps) => {
    const { appendImportEntry } = useImportDatasetAsNewProject();
    const { currentStagedId, setCurrentStagedId, currentStep, setCurrentStep } = useImportDatasetDialog();

    const handleFileUploaded = (response: FileUploadedResponse) => {
        appendImportEntry({ ...response, step: 'preparing', importJobId: null });
        setCurrentStep('preparing');
        setCurrentStagedId(response.stagedDatasetId);
    };

    const handleFilePrepared = () => {
        setCurrentStep('taskTypeSelection');
    };

    return (
        <DialogContainer onDismiss={dialogState.close}>
            {dialogState.isOpen && (
                <Dialog aria-label={'import-dataset-dialog'} width={860}>
                    <Heading>Create project from a dataset - Import</Heading>
                    <Divider />
                    <Content UNSAFE_className={classes.container}>
                        <ProgressStepper currentStep={currentStep} />

                        <View flex={'1'} width={'100%'} backgroundColor={'gray-50'}>
                            {currentStep === 'uploading' && (
                                <ImportUploadFile formatOptions='Geti, COCO' onFileUploaded={handleFileUploaded} />
                            )}

                            {currentStep === 'preparing' && isNonEmptyString(currentStagedId) && (
                                <ImportProcess stagedDatasetId={currentStagedId} onFilePrepared={handleFilePrepared} />
                            )}

                            {currentStep === 'taskTypeSelection' && isNonEmptyString(currentStagedId) && (
                                <ImportTaskSelection stagedDatasetId={currentStagedId} />
                            )}
                        </View>
                    </Content>

                    <ImportDatasetButtons
                        currentStep={currentStep}
                        stagedDatasetId={currentStagedId}
                        onClose={dialogState.close}
                    />
                </Dialog>
            )}
        </DialogContainer>
    );
};
