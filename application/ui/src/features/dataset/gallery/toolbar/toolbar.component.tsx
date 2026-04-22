// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, useMemo } from 'react';

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
    ViewModes,
} from '@geti/ui';
import { isString } from 'lodash-es';

import type { Media } from '../../../../constants/shared-types';
import { isImage } from '../../../../shared/media-item-utils';
import { TrainModel } from '../../../models/train-model/train-model.component';
import { ImportExport } from '../../import-export/import-export.component';
import { useSelectedData } from '../../providers/selected-data-provider.component';
import { DeleteMediaItem } from '../delete-media-item/delete-media-item.component';
import { useSelectDatasetItem } from '../hooks/use-select-dataset-item.hook';
import { AssignLabel } from './assign-label.component';
import { DatasetStatistics } from './dataset-statistics/dataset-statistics.component';
import { FilterByStatus, type FilterByStatusKey } from './filter-by-status/filter-by-status.component';
import { MediaFilterLabels } from './media-filter-labels/media-filter-labels.component';
import { MediaUpload } from './media-upload.component';
import { getNumberOfImagesAndVideosMessage, toggleMultipleSelection } from './util';

type ToolbarProps = {
    items: Media[];
    viewMode: ViewModes;
    totalItemsCount: number;
    setViewMode: Dispatch<SetStateAction<ViewModes>>;
    onFilter: (status: FilterByStatusKey) => void;
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

export const Toolbar = ({ items, viewMode, totalItemsCount, setViewMode, onFilter }: ToolbarProps) => {
    const { onSelectedMediaItemChange } = useSelectDatasetItem();
    const { selectedKeys, setSelectedKeys, toggleSelectedKeys } = useSelectedData();

    const selectedMediaItems = selectedKeys instanceof Set ? selectedKeys : null;

    const totalSelectedElements = selectedMediaItems?.size ?? 0;
    const hasSelectedElements = totalSelectedElements > 0;
    const message = hasSelectedElements
        ? `${totalSelectedElements} selected`
        : getNumberOfImagesAndVideosMessage(items, totalItemsCount);

    const handleToggleManyItemSelection = () => {
        const images = items.map((item) => String(item.id));
        setSelectedKeys(toggleMultipleSelection(images));
    };

    const selectedImagesIds = useMemo(() => {
        if (selectedMediaItems === null) return [];

        return Array.from(selectedMediaItems)
            .filter((itemId) => items.some((item) => itemId === item.id && isImage(item)))
            .filter((itemId) => isString(itemId));
    }, [selectedMediaItems, items]);

    return (
        <Flex direction={'column'} gridArea={'toolbar'} gap={'size-200'} marginBottom={'size-200'}>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Heading level={1}>Dataset</Heading>
                <ButtonGroup UNSAFE_style={{ gap: dimensionValue('size-125') }}>
                    <ImportExport />

                    <MediaUpload />

                    <AssignLabel selectedImagesIds={selectedImagesIds} />

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

                            {/*
                                TODO: In the future we will have a single endpoint to accept/decline
                                    multiple media items at once instead of sending multiple requests in a loop.
                                    Once we have that, we can reenable these buttons.
                            */}
                            {/* <Button variant={'accent'} onPress={handleAccept}>
                                Accept
                            </Button>
                            <Button variant={'secondary'} onPress={handleReject}>
                                Decline
                            </Button> */}
                        </>
                    )}
                </Flex>

                <Flex gap={'size-200'} alignItems={'center'}>
                    <Text>{message}</Text>

                    <FilterByStatus onChange={onFilter} />

                    <MediaFilterLabels />

                    <DatasetStatistics />

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
