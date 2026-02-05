// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, Key, Picker, Tag, Text } from '@geti/ui';
import { Accept, Search } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';
import { DatasetSubset, Media } from '../../../../constants/shared-types';
import { Hotkeys } from '../primary-toolbar/hotkeys/hotkeys.component';
import { Settings } from '../primary-toolbar/settings/settings.component';
import { ToggleFocus } from '../primary-toolbar/toggle-focus.component';
import { ZoomFitScreen } from '../primary-toolbar/zoom/zoom-fit-screen.component';
import { ZoomSelector } from '../primary-toolbar/zoom/zoom-selector.component';
import { Toolbar } from '../toolbar-container/toolbar-container.component';

import classes from './bottom-toolbar.module.scss';

type BottomToolbarProps = {
    isUserReviewed: boolean;
    mediaItem: Media;
};

const useUpdateSubset = () => {
    const projectId = useProjectIdentifier();

    const updateSubsetMutation = $api.useMutation(
        'patch',
        '/api/projects/{project_id}/dataset/items/{dataset_item_id}/subset'
    );

    const handleSubsetChange = (key: Key | null, mediaItem: Media) => {
        const subset = key as Exclude<DatasetSubset, 'unassigned'>;

        updateSubsetMutation.mutate({
            params: {
                path: {
                    project_id: projectId,
                    dataset_item_id: mediaItem.id,
                },
            },
            body: {
                subset,
            },
        });
    };

    return { handleSubsetChange };
};

export const BottomToolbar = ({ isUserReviewed, mediaItem }: BottomToolbarProps) => {
    const fileName = `${mediaItem.name}.${mediaItem.format} (${mediaItem.width} x ${mediaItem.height} px)`;

    const { handleSubsetChange } = useUpdateSubset();

    return (
        <Flex justifyContent={'end'}>
            <Toolbar.Container>
                <Grid autoFlow={'column'} autoColumns={'max-content'} gap={'size-50'}>
                    <Toolbar.Section>
                        <Hotkeys />
                    </Toolbar.Section>

                    <Toolbar.Section>
                        <Flex gap={'size-100'} alignItems={'center'}>
                            <Text UNSAFE_className={classes.filename}>{fileName}</Text>
                            <Tag
                                className={clsx({
                                    [classes.accepted]: isUserReviewed,
                                    [classes.forReview]: !isUserReviewed,
                                })}
                                prefix={isUserReviewed ? <Accept /> : <Search />}
                                text={isUserReviewed ? 'Accepted' : 'For Review'}
                            />

                            <Picker
                                placeholder={'Select subset'}
                                aria-label={'Select subset'}
                                onSelectionChange={(key) => handleSubsetChange(key, mediaItem)}
                            >
                                <Item key={'validation'}>Validation</Item>
                                <Item key={'testing'}>Testing</Item>
                                <Item key={'training'}>Training</Item>
                            </Picker>
                        </Flex>
                    </Toolbar.Section>

                    <Toolbar.Section>
                        <Flex alignItems={'center'}>
                            <Settings />

                            <ZoomSelector />

                            <ToggleFocus />

                            <ZoomFitScreen />
                        </Flex>
                    </Toolbar.Section>
                </Grid>
            </Toolbar.Container>
        </Flex>
    );
};
