// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ChangeEvent, useRef } from 'react';

import { Button } from '@geti/ui';

type AddMediaButtonProps = {
    onFilesSelected: (files: File[]) => void;
    multiple?: boolean;
};

const VALID_VIDEO_EXT = ['mp4', 'avi', 'mkv', 'mov', 'webm', 'm4v'];
const VALID_IMAGE_EXT = ['jpg', 'jpeg', 'png', 'jfif', 'tif', 'tiff', 'webp', 'bmp'];
const VALID_EXT = [...VALID_VIDEO_EXT, ...VALID_IMAGE_EXT];

export const acceptedExtensions = VALID_EXT.map((ext) => `.${ext}`).join(',');

export const AddMediaButton = ({ onFilesSelected, multiple = true }: AddMediaButtonProps) => {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files;

        if (files && files.length > 0) {
            const fileArray = Array.from(files);
            onFilesSelected(fileArray);
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
                multiple={multiple}
                onChange={handleFileChange}
                style={{ display: 'none' }}
                aria-label={'Upload media files'}
                accept={acceptedExtensions}
            />
            <Button variant={'secondary'} onPress={handleClick}>
                Upload Files
            </Button>
        </>
    );
};
