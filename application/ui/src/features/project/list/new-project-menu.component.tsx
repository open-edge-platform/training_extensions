// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import { ActionButton, Item, Menu, MenuTrigger, Text } from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { useNavigate } from 'react-router-dom';

import { paths } from '../../../constants/paths';
import { useImportDatasetDialog } from '../providers/import-dataset-dialog-provider.component';
import { ImportDatasetAsNewProject } from './import-dataset-as-new-project/import-dataset-as-new-project.component';

import classes from './new-project-menu.module.scss';

export const NewProjectMenu = () => {
    const navigate = useNavigate();
    const { datasetImportDialogState, setCurrentStep, setCurrentStagedId } = useImportDatasetDialog();

    const handleAction = (key: Key) => {
        switch (key) {
            case 'newEmptyProject':
                navigate(paths.project.new.pattern);
                break;

            case 'newFromDataset':
                setCurrentStep('uploading');
                setCurrentStagedId(null);
                datasetImportDialogState.open();
                break;
        }
    };

    return (
        <>
            <MenuTrigger direction='bottom' align='end'>
                <ActionButton UNSAFE_className={classes.menu}>
                    <AddCircle />

                    <Text UNSAFE_className={classes.create}>Create project</Text>
                </ActionButton>
                <Menu onAction={handleAction}>
                    <Item key='newEmptyProject'>Create new project</Item>
                    <Item key='newFromDataset'>Create from dataset</Item>
                </Menu>
            </MenuTrigger>

            <ImportDatasetAsNewProject dialogState={datasetImportDialogState} />
        </>
    );
};
