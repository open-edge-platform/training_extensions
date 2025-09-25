// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ChangeEvent, useRef } from 'react';

import { Button } from '@geti/ui';

interface AddMediaButtonProps {
    onFilesSelected: (files: File[]) => void;
    multiple?: boolean;
}

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
            />
            <Button onPress={handleClick}>Upload Files</Button>
        </>
    );
};
