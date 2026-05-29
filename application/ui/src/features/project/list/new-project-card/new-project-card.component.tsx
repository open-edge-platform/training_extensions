// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Text, View } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { useNavigate } from 'react-router-dom';

import { paths } from '../../../../constants/paths';
import { useImportDatasetDialog } from '../../providers/import-dataset-dialog-provider.component';
import { ImportDatasetAsNewProject } from '../import-dataset-as-new-project/import-dataset-as-new-project.component';

import classes from './new-project-menu.module.scss';

export const NewProjectCard = () => {
    const navigate = useNavigate();
    const { datasetImportDialogState, setCurrentStep, setCurrentStagedId } = useImportDatasetDialog();

    const handleCreateProject = () => {
        navigate(paths.project.new.pattern);
    };

    const handleCreateFromDataset = () => {
        setCurrentStep('uploading');
        setCurrentStagedId(null);
        datasetImportDialogState.open();
    };

    return (
        <>
            <View UNSAFE_className={classes.card}>
                <ActionButton onPress={handleCreateProject} UNSAFE_className={classes.buttonText}>
                    <AddCircle />
                    <Text>
                        Create
                        <br />
                        new project
                    </Text>
                </ActionButton>
                <ActionButton onPress={handleCreateFromDataset} UNSAFE_className={classes.buttonText}>
                    <AddCircle />
                    <Text>
                        Create
                        <br />
                        project from dataset
                    </Text>
                </ActionButton>
            </View>
            <ImportDatasetAsNewProject dialogState={datasetImportDialogState} />
        </>
    );
};
