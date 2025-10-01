// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen, waitFor } from 'test-utils/render';

import { ImagesFolderSourceConfig } from '../util';
import { ImageFolder } from './image-folder.component';
import { useActionImageFolder } from './use-action-image-folder.hook';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

vi.mock('./use-action-image-folder.hook');

const mockedConfig: ImagesFolderSourceConfig = {
    id: '1',
    name: 'Test Folder',
    source_type: 'images_folder',
    images_folder_path: '/path/to/images',
    ignore_existing_images: false,
};

describe('ImageFolder', () => {
    it('disables the Apply button when loading', () => {
        vi.mocked(useActionImageFolder).mockReturnValue([mockedConfig, vi.fn(), true]);

        render(<ImageFolder />);

        expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
    });

    it('calls submit action when Apply button is clicked', () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useActionImageFolder).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(<ImageFolder />);

        fireEvent.click(screen.getByRole('button', { name: 'Apply' }));

        waitFor(() => {
            expect(mockedSubmitAction).toHaveBeenCalled();
        });
    });
});
