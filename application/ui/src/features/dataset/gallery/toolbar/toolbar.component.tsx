// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, SetStateAction, Suspense } from 'react';

import {
    Button,
    ButtonGroup,
    Checkbox,
    dimensionValue,
    Divider,
    Flex,
    Heading,
    Loading,
    MediaViewModes,
    Text,
    ViewModes,
} from '@geti/ui';

import type { Media } from '../../../../constants/shared-types';
import { TrainModel } from '../../../models/train-model/train-model.component';
import { ImportExport } from '../../import-export/import-export.component';
import { useSelectedData } from '../../providers/selected-data-provider.component';
import { BulkLabelsAssignmentDialog } from '../bulk-labels-assignment/bulk-labels-assignment-dialog.component';
import { useBulkUploadAndAssignLabel } from '../bulk-labels-assignment/use-bulk-upload-and-assign-label';
import { DeleteMediaItem } from '../delete-media-item/delete-media-item.component';
import { useSelectDatasetItem } from '../hooks/use-select-dataset-item.hook';
import { AddMediaButton } from './add-media-button/add-media-button.component';
import { DatasetStatistics } from './dataset-statistics/dataset-statistics.component';
import { FilterByStatus, type FilterByStatusKey } from './filter-by-status/filter-by-status.component';
import { toggleMultipleSelection } from './util';

type ToolbarProps = {
    items: Media[];
    viewMode: ViewModes;
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

const MediaUpload = () => {
    const {
        isClassification,
        isMultiLabelClassification,
        setFilesForLabelAssignment,
        filesForLabelAssignment,
        uploadAndAssign,
        uploadMedia,
        uploadMediaLoading,
    } = useBulkUploadAndAssignLabel();

    const handleClose = () => {
        setFilesForLabelAssignment([]);
    };

    return (
        <>
            <AddMediaButton onFileUpload={uploadAndAssign} isDisabled={uploadMediaLoading} />
            {isClassification && (
                <BulkLabelsAssignmentDialog
                    onClose={handleClose}
                    files={filesForLabelAssignment}
                    onDatasetItemsUpload={uploadMedia}
                    isUploadingDatasetItems={uploadMediaLoading}
                    isMultiLabelClassification={isMultiLabelClassification}
                />
            )}
        </>
    );
};

export const Toolbar = ({ items, viewMode, setViewMode, onFilter }: ToolbarProps) => {
    const { onSelectedMediaItemChange } = useSelectDatasetItem();
    const { selectedKeys, setSelectedKeys, toggleSelectedKeys } = useSelectedData();

    const totalSelectedElements = selectedKeys instanceof Set ? selectedKeys.size : 0;
    const hasSelectedElements = totalSelectedElements > 0;
    const message = hasSelectedElements ? `${totalSelectedElements} selected` : `${items.length} images`;

    const handleToggleManyItemSelection = () => {
        const images = items.map((item) => String(item.id));
        setSelectedKeys(toggleMultipleSelection(images));
    };

    return (
        <Flex direction={'column'} gridArea={'toolbar'} gap={'size-200'} marginBottom={'size-200'}>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Heading level={1}>Dataset</Heading>
                <ButtonGroup UNSAFE_style={{ gap: dimensionValue('size-125') }}>
                    <ImportExport />

                    <MediaUpload />

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
                    <FilterByStatus onChange={onFilter} />
                    <Text>{message}</Text>

                    <Suspense fallback={<Loading size='S' mode='inline' />}>
                        <DatasetStatistics />
                    </Suspense>

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
