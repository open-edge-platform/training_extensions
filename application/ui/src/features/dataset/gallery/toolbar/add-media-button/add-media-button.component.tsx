// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ChangeEvent, useRef } from 'react';

import { Button } from '@geti/ui';

import { VALID_EXT } from '../../utils';

export const acceptedExtensions = VALID_EXT.map((ext) => `.${ext}`).join(',');

type AddMediaButtonProps = {
    onFileUpload: (files: File[]) => Promise<void>;
    isDisabled?: boolean;
};

export const AddMediaButton = ({ onFileUpload, isDisabled = false }: AddMediaButtonProps) => {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files;

        if (files && files.length > 0) {
            const fileArray = Array.from(files);

            await onFileUpload(fileArray);
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
        </>
    );
};
