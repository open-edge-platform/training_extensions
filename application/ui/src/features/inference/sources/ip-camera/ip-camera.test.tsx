// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from '@test-utils/render';
import userEvent from '@testing-library/user-event';

import { useSourceAction } from '../hooks/use-source-action.hook';
import { ImagesFolderSourceConfig } from '../util';
import { IpCamera } from './ip-camera.component';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

vi.mock('../hooks/use-source-action.hook');

const mockedConfig: ImagesFolderSourceConfig = {
    id: '1',
    name: 'Test Folder',
    source_type: 'images_folder',
    images_folder_path: '/path/to/images',
    ignore_existing_images: false,
};

describe('IpCamera', () => {
    it('disables the Apply button when loading', () => {
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, vi.fn(), true]);

        render(<IpCamera />);

        expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
    });

    it('calls submit action when Apply button is clicked', () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSourceAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(<IpCamera />);

        userEvent.click(screen.getByRole('button', { name: 'Apply' }));

        waitFor(() => {
            expect(mockedSubmitAction).toHaveBeenCalled();
        });
    });
});
