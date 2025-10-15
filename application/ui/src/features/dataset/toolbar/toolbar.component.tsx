// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Divider, Flex, Heading, Text, toast } from '@geti/ui';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../api/client';
import { AddMediaButton } from '../../../components/add-media-button/add-media-button.component';
import { CheckboxInput } from '../../../components/checkbox-input/checkbox-input.component';
import { useSelectedData } from '../../../routes/dataset/provider';
import { DatasetItem } from '../../annotator/types';
import { DeleteMediaItem } from '../gallery/delete-media-item/delete-media-item.component';
import { toggleMultipleSelection, updateSelectedKeysTo } from './util';

type ToolbarProps = {
    items: DatasetItem[];
};

export const Toolbar = ({ items }: ToolbarProps) => {
    const projectId = useProjectIdentifier();
    const { selectedKeys, setSelectedKeys, setMediaState, toggleSelectedKeys } = useSelectedData();

    const addItemMutation = $api.useMutation('post', '/api/projects/{project_id}/dataset/items', {
        meta: {
            invalidateQueries: [['get', '/api/projects/{project_id}/dataset/items']],
        },
    });

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

    const handleAddMediaItem = (files: File[]) => {
        files.forEach((file) => {
            const formData = new FormData();
            formData.append('file', file);

            addItemMutation.mutate(
                {
                    params: { path: { project_id: projectId } },
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    body: formData as any,
                },
                {
                    onSuccess: () => {
                        toast({ type: 'success', message: `Uploaded ${files.length} item(s)` });
                    },
                }
            );
        });
    };

    return (
        <Flex direction={'column'} gridArea={'toolbar'} gap={'size-200'} marginBottom={'size-200'}>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Heading level={1}>Data collection</Heading>
                <AddMediaButton onFilesSelected={handleAddMediaItem} />
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
