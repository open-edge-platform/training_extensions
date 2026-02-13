// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction } from 'react';

import {
    Button,
    ButtonGroup,
    Checkbox,
    dimensionValue,
    Divider,
    Flex,
    Heading,
    MediaViewModes,
    Text,
    toast,
    ViewModes,
} from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import { AddMediaButton } from '../../../../components/add-media-button/add-media-button.component';
import type { Media } from '../../../../constants/shared-types';
import { getQueryKey } from '../../../../query-client/query-client';
import { TrainModel } from '../../../models/train-model/train-model.component';
import { DeleteMediaItem } from '../../gallery/delete-media-item/delete-media-item.component';
import { ImportExport } from '../../import-export/import-export.component';
import { useSelectedData } from '../../selected-data-provider.component';
import { useSelectDatasetItem } from '../hooks/use-select-dataset-item.hook';
import { toggleMultipleSelection, updateSelectedKeysTo } from './util';

type ToolbarProps = {
    items: Media[];
    viewMode: ViewModes;
    setViewMode: Dispatch<SetStateAction<ViewModes>>;
};

type AnnotateButtonProps = {
    isDisabled?: boolean;
    onClick?: () => void;
};

const AnnotateButton = ({ isDisabled, onClick }: AnnotateButtonProps) => {
    return (
        <Button margin={0} variant={'primary'} onPress={onClick} isDisabled={isDisabled}>
            Annotate
        </Button>
    );
};

export const Toolbar = ({ items, viewMode, setViewMode }: ToolbarProps) => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();

    const { onSelectedMediaItemChange } = useSelectDatasetItem();
    const { selectedKeys, setSelectedKeys, setMediaState, toggleSelectedKeys } = useSelectedData();

    const addItemMutation = $api.useMutation('post', '/api/projects/{project_id}/dataset/media');

    const totalSelectedElements = selectedKeys instanceof Set ? selectedKeys.size : 0;
    const hasSelectedElements = totalSelectedElements > 0;
    const message = hasSelectedElements ? `${totalSelectedElements} selected` : `${items.length} images`;

    const handleToggleManyItemSelection = () => {
        const images = items.map((item) => String(item.id));
        setSelectedKeys(toggleMultipleSelection(images));
    };

    const handleAccept = () => {
        setSelectedKeys(new Set());
        setMediaState(updateSelectedKeysTo(selectedKeys, 'accepted'));
    };

    const handleReject = () => {
        setSelectedKeys(new Set());
        setMediaState(updateSelectedKeysTo(selectedKeys, 'rejected'));
    };

    const handleAddMediaItem = async (files: File[]) => {
        const uploadPromises = files.map((file) => {
            const formData = new FormData();
            formData.append('file', file);

            return addItemMutation.mutateAsync({
                params: { path: { project_id: projectId } },
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                body: formData as any,
            });
        });

        const promises = await Promise.allSettled(uploadPromises);

        const succeeded = promises.filter((result) => result.status === 'fulfilled').length;
        const failed = promises.filter((result) => result.status === 'rejected').length;

        await queryClient.invalidateQueries({
            queryKey: getQueryKey([
                'get',
                '/api/projects/{project_id}/dataset/media',
                {
                    params: {
                        path: {
                            project_id: projectId,
                        },
                    },
                },
            ]),
        });

        if (failed === 0) {
            toast({ type: 'success', message: `Uploaded ${succeeded} item(s)` });
        } else if (succeeded === 0) {
            toast({ type: 'error', message: `Failed to upload ${failed} item(s)` });
        } else {
            toast({
                type: 'warning',
                message: `Uploaded ${succeeded} item(s), ${failed} failed`,
            });
        }
    };

    return (
        <Flex direction={'column'} gridArea={'toolbar'} gap={'size-200'} marginBottom={'size-200'}>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Heading level={1}>Dataset</Heading>
                <ButtonGroup UNSAFE_style={{ gap: dimensionValue('size-125') }}>
                    <ImportExport />

                    <AddMediaButton onFilesSelected={handleAddMediaItem} />

                    <TrainModel />

                    <AnnotateButton
                        isDisabled={items.at(0) === undefined}
                        onClick={items.at(0) === undefined ? undefined : () => onSelectedMediaItemChange(items[0])}
                    />
                </ButtonGroup>
            </Flex>

            <Divider size='S' />

            <Flex direction={'row'} alignItems={'center'} justifyContent={'space-between'}>
                <Flex
                    gap={'size-200'}
                    height={'size-400'}
                    direction={'row'}
                    alignItems={'center'}
                    justifyContent={'space-between'}
                >
                    <Checkbox
                        aria-label={'select all'}
                        onChange={handleToggleManyItemSelection}
                        isSelected={hasSelectedElements && totalSelectedElements === items.length}
                    />

                    <Divider orientation={'vertical'} size={'S'} />

                    {hasSelectedElements && (
                        <>
                            <DeleteMediaItem
                                itemsIds={Array.from(selectedKeys) as string[]}
                                onDeleted={toggleSelectedKeys}
                            />

                            <Button variant={'accent'} onPress={handleAccept}>
                                Accept
                            </Button>
                            <Button variant={'secondary'} onPress={handleReject}>
                                Decline
                            </Button>
                        </>
                    )}
                </Flex>

                <Flex gap={'size-200'} alignItems={'center'}>
                    <Text>{message}</Text>
                    <MediaViewModes
                        viewMode={viewMode}
                        setViewMode={setViewMode}
                        items={[ViewModes.LARGE, ViewModes.MEDIUM, ViewModes.SMALL]}
                    />
                </Flex>
            </Flex>

            <Divider size='S' />
        </Flex>
    );
};
