// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ChangeEvent, useRef, useState } from 'react';

import { Button } from '@geti/ui';

import { Project } from '../../../../../constants/shared-types';
import { isClassificationTask, isMultiLabelClassificationTask } from '../../../../project/task-type-guards';
import { BulkLabelAssignmentDialog } from './bulk-label-assignment/bulk-label-assignment-dialog.component';

const VALID_VIDEO_EXT = ['mp4', 'avi', 'mkv', 'mov', 'webm', 'm4v'];
const VALID_IMAGE_EXT = ['jpg', 'jpeg', 'png', 'jfif', 'tif', 'tiff', 'webp', 'bmp'];
const VALID_EXT = [...VALID_VIDEO_EXT, ...VALID_IMAGE_EXT];

export const acceptedExtensions = VALID_EXT.map((ext) => `.${ext}`).join(',');

type AddMediaButtonProps = {
    onFilesSelected: (files: File[]) => Promise<void>;
    isDisabled?: boolean;
    project: Project;
};

export const AddMediaButton = ({ onFilesSelected, isDisabled = false, project }: AddMediaButtonProps) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [filesForLabelAssignment, setFilesForLabelAssignment] = useState<File[]>([]);

    const isClassification = isClassificationTask(project.task.task_type);
    const isMultiLabelClassification = isMultiLabelClassificationTask(project.task);

    const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files;

        if (files && files.length > 0) {
            const fileArray = Array.from(files);

            if (isClassification) {
                setFilesForLabelAssignment(fileArray);
            } else {
                await onFilesSelected(fileArray);
            }
        }

        // Clear the input value to allow selecting the same file again
        if (event.target) {
            event.target.value = '';
        }
    };

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    return (
        <>
            <input
                ref={fileInputRef}
                type='file'
                multiple
                onChange={handleFileChange}
                style={{ display: 'none' }}
                aria-label={'Upload media files'}
                accept={acceptedExtensions}
            />
            <Button variant={'secondary'} isDisabled={isDisabled} onPress={handleClick}>
                Upload media
            </Button>
            {isClassification && (
                <BulkLabelAssignmentDialog
                    files={filesForLabelAssignment}
                    onClose={() => setFilesForLabelAssignment([])}
                    onDatasetItemsUpload={onFilesSelected}
                    isMultiLabelClassification={isMultiLabelClassification}
                />
            )}
        </>
    );
};
