// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, ButtonGroup, Divider, Flex, Heading, Text, toast } from '@geti/ui';
import { useQueryClient } from '@tanstack/react-query';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { DatasetItem } from 'src/constants/shared-types';

import { $api } from '../../../../api/client';
import { AddMediaButton } from '../../../../components/add-media-button/add-media-button.component';
import { CheckboxInput } from '../../../../components/checkbox-input/checkbox-input.component';
import { TrainModel } from '../../../models/train-model/train-model';
import { DeleteMediaItem } from '../../gallery/delete-media-item/delete-media-item.component';
import { useSelectedData } from '../../selected-data-provider.component';
import { toggleMultipleSelection, updateSelectedKeysTo } from './util';

type ToolbarProps = {
    items: DatasetItem[];
};

export const Toolbar = ({ items }: ToolbarProps) => {
    const projectId = useProjectIdentifier();
    const queryClient = useQueryClient();
    const { selectedKeys, setSelectedKeys, setMediaState, toggleSelectedKeys } = useSelectedData();

    const addItemMutation = $api.useMutation('post', '/api/projects/{project_id}/dataset/items');

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
            queryKey: ['get', '/api/projects/{project_id}/dataset/items'],
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
                <Heading level={1}>Data collection</Heading>
                <ButtonGroup>
                    <AddMediaButton onFilesSelected={handleAddMediaItem} />
                    <TrainModel />
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
                    <CheckboxInput
                        name={'select all'}
                        onChange={handleToggleManyItemSelection}
                        isChecked={totalSelectedElements === items.length}
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

                <Text>{message}</Text>
            </Flex>

            <Divider size='S' />
        </Flex>
    );
};
