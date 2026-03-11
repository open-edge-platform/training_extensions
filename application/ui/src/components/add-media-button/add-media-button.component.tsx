// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ChangeEvent, useRef } from 'react';

import { Button, Item, Key, Menu, MenuTrigger } from '@geti/ui';

type AddMediaButtonProps = {
    onFilesSelected: (files: File[]) => void;
    isDisabled?: boolean;
};

const VALID_VIDEO_EXT = ['mp4', 'avi', 'mkv', 'mov', 'webm', 'm4v'];
const VALID_IMAGE_EXT = ['jpg', 'jpeg', 'png', 'jfif', 'tif', 'tiff', 'webp', 'bmp'];
const VALID_EXT = [...VALID_VIDEO_EXT, ...VALID_IMAGE_EXT];

export const acceptedExtensions = VALID_EXT.map((ext) => `.${ext}`).join(',');

export const AddMediaButton = ({ onFilesSelected, isDisabled = false }: AddMediaButtonProps) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const folderInputRef = useRef<HTMLInputElement>(null);

    const setFolderInputRef = (input: HTMLInputElement | null) => {
        folderInputRef.current = input;

        if (input === null) {
            return;
        }

        input.setAttribute('webkitdirectory', '');
        input.setAttribute('directory', '');
    };

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

    const handleMenuAction = (action: Key) => {
        if (action === 'files') {
            fileInputRef.current?.click();
        } else if (action === 'folder') {
            folderInputRef.current?.click();
        }
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
            <input
                ref={setFolderInputRef}
                type='file'
                multiple
                onChange={handleFileChange}
                style={{ display: 'none' }}
                aria-label={'Upload media folder'}
                accept={acceptedExtensions}
            />
            <MenuTrigger>
                <Button variant={'secondary'} isDisabled={isDisabled}>
                    Upload media
                </Button>
                <Menu onAction={handleMenuAction} aria-label={'Upload menu'}>
                    <Item key={'files'}>Files</Item>
                    <Item key={'folder'}>Folder</Item>
                </Menu>
            </MenuTrigger>
        </>
    );
};
