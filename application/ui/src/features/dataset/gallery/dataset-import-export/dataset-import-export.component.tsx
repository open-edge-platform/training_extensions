// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Item, Key, Menu, MenuTrigger } from '@geti/ui';
import { useOverlayTriggerState } from '@react-stately/overlays';

import { ExportDataset } from '../export-dataset/export-dataset.component';

export const DatasetImportExport = () => {
    const dialogState = useOverlayTriggerState({});

    const handleMenuAction = (option: Key) => {
        switch (option) {
            case 'export':
                dialogState.open();
                break;
            case 'import':
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
                    {/* Todo: coming after export */}
                    {/* <Item key='import'>Import dataset</Item> */}
                </Menu>
            </MenuTrigger>

            <ExportDataset dialogState={dialogState} />
        </>
    );
};
