// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti/ui';
import { useNavigate } from 'react-router';

import { ReactComponent as EmptyFolderImage } from '../../../../assets/empty-folder.svg';
import { paths } from '../../../../constants/paths';
import { useImportDatasetDialog } from '../../providers/import-dataset-dialog-provider.component';
import { CreateProjectButton } from '../create-project-button/create-project-button.component';
import { ImportDatasetAsNewProject } from '../import-dataset-as-new-project/import-dataset-as-new-project.component';

import classes from './empty-project-list.module.scss';

export const EmptyProjectList = () => {
    const navigate = useNavigate();
    const { datasetImportDialogState, setCurrentStep, setCurrentStagedId } = useImportDatasetDialog();

    const handleCreateFromDataset = () => {
        setCurrentStep('uploading');
        setCurrentStagedId(null);
        datasetImportDialogState.open();
    };

    return (
        <Flex
            gap={'size-100'}
            direction={'column'}
            alignItems={'center'}
            justifyContent={'center'}
            UNSAFE_className={classes.container}
        >
            <EmptyFolderImage aria-label='empty list' />

            <CreateProjectButton
                buttonText='Create Project'
                handleOpenDialog={() => navigate(paths.project.new.pattern)}
                onCreateFromDataset={handleCreateFromDataset}
            />
            <ImportDatasetAsNewProject dialogState={datasetImportDialogState} />
        </Flex>
    );
};
