// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key } from 'react';

import { Button, Item, Menu, MenuTrigger as SpectrumMenuTrigger, Tooltip, TooltipTrigger } from '@geti/ui';
import { ChevronDownSmall } from '@geti/ui/icons';

import classes from './create-project-button.module.scss';

export enum CreateProjectMenuActions {
    IMPORT_DATASET = 'Create from dataset',
}

interface CreateProjectMenuProps {
    onCreateFromDataset: () => void;
}

// More options will be added later, e.g. import project
const items = [{ name: CreateProjectMenuActions.IMPORT_DATASET, id: CreateProjectMenuActions.IMPORT_DATASET }];

export const CreateProjectMenu = ({ onCreateFromDataset }: CreateProjectMenuProps) => {
    const onOpenDialog = (key: Key) => {
        switch (key) {
            case CreateProjectMenuActions.IMPORT_DATASET:
                onCreateFromDataset();
                break;
        }
    };

    return (
        <>
            <SpectrumMenuTrigger>
                <TooltipTrigger placement={'bottom'}>
                    <Button
                        variant={'accent'}
                        id={'create-project-menu'}
                        aria-label={'Create project menu'}
                        UNSAFE_className={classes.createProjectMenuButton}
                        minWidth={'size-450'}
                    >
                        <ChevronDownSmall />
                    </Button>
                    <Tooltip>Create project menu</Tooltip>
                </TooltipTrigger>

                <Menu items={items} onAction={onOpenDialog}>
                    {(item) => <Item>{item.name}</Item>}
                </Menu>
            </SpectrumMenuTrigger>
        </>
    );
};
