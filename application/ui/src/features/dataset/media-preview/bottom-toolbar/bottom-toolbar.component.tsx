// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Grid, Item, Key, Picker, Tag, Text } from '@geti/ui';
import { Accept, Search } from '@geti/ui/icons';
import { clsx } from 'clsx';
import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';
import { capitalize } from 'lodash-es';

import { $api } from '../../../../api/client';
import { DatasetSubset, Media } from '../../../../constants/shared-types';
import { getQueryKey } from '../../../../query-client/query-client';
import { Hotkeys } from '../primary-toolbar/hotkeys/hotkeys.component';
import { Settings } from '../primary-toolbar/settings/settings.component';
import { ToggleFocus } from '../primary-toolbar/toggle-focus.component';
import { ZoomFitScreen } from '../primary-toolbar/zoom/zoom-fit-screen.component';
import { ZoomSelector } from '../primary-toolbar/zoom/zoom-selector.component';
import { Toolbar } from '../toolbar-container/toolbar-container.component';

import classes from './bottom-toolbar.module.scss';

type BottomToolbarProps = {
    mediaItem: Media;
    hideHotkeys?: boolean;
};

const DATASET_ITEM_OPERATION = 'get';
const DATASET_ITEM_URL = '/api/projects/{project_id}/dataset/items/{dataset_item_id}';
const UPDATE_SUBSET_OPERATION = 'patch';
const UPDATE_SUBSET_URL = '/api/projects/{project_id}/dataset/items/{dataset_item_id}/subset';

type AssignableSubset = Exclude<DatasetSubset, 'unassigned'>;

const isAssignableSubset = (key: Key | null): key is AssignableSubset => key !== null && key !== 'unassigned';

const useSubsets = (mediaItemId: string) => {
    const projectId = useProjectIdentifier();

    const datasetItemParamsPath = {
        project_id: projectId,
        dataset_item_id: mediaItemId,
    };
    const datasetItemParams = { params: { path: datasetItemParamsPath } };
    const mediaListParams = { params: { path: { project_id: projectId } } };

    const { data } = $api.useQuery(DATASET_ITEM_OPERATION, DATASET_ITEM_URL, datasetItemParams);

    const updateSubsetMutation = $api.useMutation(UPDATE_SUBSET_OPERATION, UPDATE_SUBSET_URL, {
        meta: {
            invalidateQueries: [
                getQueryKey([DATASET_ITEM_OPERATION, DATASET_ITEM_URL, datasetItemParams]),
                getQueryKey(['get', '/api/projects/{project_id}/dataset/media', mediaListParams]),
            ],
        },
    });

    const handleSubsetChange = (key: Key | null) => {
        if (!isAssignableSubset(key)) return;

        updateSubsetMutation.mutate({
            params: { path: datasetItemParamsPath },
            body: { subset: key },
        });
    };

    return { currentSubset: data?.subset ?? null, isUserReviewed: data?.user_reviewed ?? false, handleSubsetChange };
};

export const BottomToolbar = ({ mediaItem, hideHotkeys }: BottomToolbarProps) => {
    const fileName = `${mediaItem.name}.${mediaItem.format} (${mediaItem.width} x ${mediaItem.height} px)`;

    const { currentSubset, isUserReviewed, handleSubsetChange } = useSubsets(mediaItem.id);

    const isUnassigned = currentSubset === 'unassigned' || currentSubset === null;

    return (
        <Flex justifyContent={'end'}>
            <Toolbar.Container>
                <Grid autoFlow={'column'} autoColumns={'max-content'} gap={'size-50'}>
                    {!hideHotkeys && (
                        <Toolbar.Section>
                            <Hotkeys />
                        </Toolbar.Section>
                    )}

                    <Toolbar.Section>
                        <Flex gap={'size-100'} alignItems={'center'} height={'100%'}>
                            <Text UNSAFE_className={classes.filename}>{fileName}</Text>
                            <Tag
                                className={clsx({
                                    [classes.accepted]: isUserReviewed,
                                    [classes.forReview]: !isUserReviewed,
                                })}
                                prefix={isUserReviewed ? <Accept /> : <Search />}
                                text={isUserReviewed ? 'Accepted' : 'For Review'}
                            />

                            {isUnassigned ? (
                                <Picker
                                    selectedKey={null}
                                    placeholder={'Select subset'}
                                    aria-label={'Select subset'}
                                    onSelectionChange={handleSubsetChange}
                                >
                                    <Item key={'validation'}>Validation</Item>
                                    <Item key={'testing'}>Testing</Item>
                                    <Item key={'training'}>Training</Item>
                                </Picker>
                            ) : (
                                <Tag withDot={false} text={capitalize(String(currentSubset))} />
                            )}
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
