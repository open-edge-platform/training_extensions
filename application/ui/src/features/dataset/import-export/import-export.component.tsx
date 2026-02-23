// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Item, Key, Menu, MenuTrigger } from '@geti/ui';
import { useOverlayTriggerState } from '@react-stately/overlays';

import { ExportDataset } from './export-dataset/export-dataset.component';
import { ImportDataset } from './import-dataset/import-dataset.component';

export const ImportExport = () => {
    const exportDialogState = useOverlayTriggerState({});
    const importDialogState = useOverlayTriggerState({});

    const handleMenuAction = (option: Key) => {
        switch (option) {
            case 'export':
                exportDialogState.open();
                break;
            case 'import':
                importDialogState.open();
                break;
        }
    };
    return (
        <>
            <MenuTrigger>
                <Button variant='secondary' aria-label='import-export dataset'>
                    Export/Import
                </Button>
                <Menu onAction={handleMenuAction}>
                    <Item key='export'>Export dataset</Item>
                    <Item key='import'>Import dataset</Item>
                </Menu>
            </MenuTrigger>

            <ExportDataset dialogState={exportDialogState} />
            <ImportDataset dialogState={importDialogState} />
        </>
    );
};
