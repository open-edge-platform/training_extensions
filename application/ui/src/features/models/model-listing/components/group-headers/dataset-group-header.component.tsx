// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import {
    ActionButton,
    AlertDialog,
    DialogContainer,
    dimensionValue,
    Flex,
    Grid,
    Heading,
    Item,
    Key,
    Menu,
    MenuTrigger,
    Text,
} from '@geti/ui';
import { Image, MoreMenu, Tag } from '@geti/ui/icons';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { TrainModel } from '../../../train-model/train-model.component';
import { useDeleteDatasetRevision } from '../../hooks/use-delete-dataset-revision.hook';
import { useRenameDatasetRevision } from '../../hooks/use-rename-dataset-revision.hook';
import type { DatasetGroup } from '../../types';
import { ThreeSectionRange } from '../three-section-range/three-section-range.component';
import { RenameDatasetRevisionDialog } from './rename-dataset-revision-dialog.component';

import classes from './group-headers.module.scss';

type DatasetGroupHeaderProps = {
    dataset: DatasetGroup;
};

export const DatasetGroupHeader = ({ dataset }: DatasetGroupHeaderProps) => {
    const projectId = useProjectIdentifier();
    const renameDatasetRevisionMutation = useRenameDatasetRevision();
    const deleteDatasetRevisionMutation = useDeleteDatasetRevision();

    const [isRenameDialogOpen, setIsRenameDialogOpen] = useState(false);
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

    const handleDatasetMenuAction = (key: Key) => {
        switch (key) {
            case 'rename':
                setIsRenameDialogOpen(true);
                break;
            case 'delete':
                setIsDeleteDialogOpen(true);
                break;
            default:
                break;
        }
    };

    const handleRename = (newName: string) => {
        renameDatasetRevisionMutation.mutate(
            {
                params: { path: { project_id: projectId, dataset_revision_id: dataset.id } },
                body: { name: newName },
            },
            {
                onSuccess: () => {
                    setIsRenameDialogOpen(false);
                },
            }
        );
    };

    const handleDelete = () => {
        deleteDatasetRevisionMutation.mutate({
            params: { path: { project_id: projectId, dataset_revision_id: dataset.id } },
        });
    };

    return (
        <Grid
            columns={['auto', '1fr', 'auto', '1fr', 'auto']}
            alignItems={'center'}
            marginBottom={'size-225'}
            gap={'size-200'}
        >
            <Flex alignItems={'center'} gap={'size-50'}>
                <Heading level={2} UNSAFE_style={{ fontSize: dimensionValue('size-300') }}>
                    {dataset.name}
                </Heading>

                <MenuTrigger>
                    <ActionButton isQuiet aria-label={'Dataset actions'}>
                        <MoreMenu />
                    </ActionButton>
                    <Menu onAction={handleDatasetMenuAction} aria-label={'Dataset actions menu'}>
                        <Item key={'rename'}>Rename</Item>
                        <Item key={'delete'}>Delete</Item>
                    </Menu>
                </MenuTrigger>
            </Flex>
            <Text
                UNSAFE_style={{
                    color: 'var(--spectrum-global-color-gray-700)',
                }}
            >
                {dataset.createdAt}
            </Text>

            <Flex gap={'size-50'} justifyContent={'center'}>
                <Flex UNSAFE_className={classes.tag}>
                    <Tag /> {dataset.labelCount}
                </Flex>
                <Flex UNSAFE_className={classes.tag}>
                    <Image /> {dataset.imageCount.toLocaleString()}
                </Flex>
            </Flex>

            <ThreeSectionRange
                trainingValue={dataset.trainingSubsets.training}
                validationValue={dataset.trainingSubsets.validation}
                testingValue={dataset.trainingSubsets.testing}
            />

            <Flex>
                <TrainModel preSelectedDatasetRevisionId={dataset.id} />
            </Flex>

            <DialogContainer onDismiss={() => setIsRenameDialogOpen(false)}>
                {isRenameDialogOpen && (
                    <RenameDatasetRevisionDialog
                        currentName={dataset.name}
                        onRename={handleRename}
                        isPending={renameDatasetRevisionMutation.isPending}
                        onClose={() => setIsRenameDialogOpen(false)}
                    />
                )}
            </DialogContainer>

            <DialogContainer onDismiss={() => setIsDeleteDialogOpen(false)}>
                {isDeleteDialogOpen && (
                    <AlertDialog
                        title='Delete'
                        variant='destructive'
                        primaryActionLabel='Delete'
                        onPrimaryAction={handleDelete}
                        cancelLabel='Cancel'
                    >
                        {`Are you sure you want to delete dataset revision "${dataset.name}"?`}
                    </AlertDialog>
                )}
            </DialogContainer>
        </Grid>
    );
};
