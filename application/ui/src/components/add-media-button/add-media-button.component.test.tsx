// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { AddMediaButton } from './add-media-button.component';

describe('AddMediaButton', () => {
    it('calls onFilesSelected correctly', () => {
        const mockOnFilesSelected = vi.fn();
        const mockFile = new File(['file content'], 'test-image.jpg', {
            type: 'image/jpeg',
            lastModified: Date.now(),
        });

        render(<AddMediaButton onFilesSelected={mockOnFilesSelected} />);

        const input = screen.getByLabelText(/Upload media files/);
        fireEvent.change(input, { target: { files: [mockFile] } });

        expect(mockOnFilesSelected).toHaveBeenCalledWith([mockFile]);
    });

    it('sets the expected accepted file extensions', () => {
        render(<AddMediaButton onFilesSelected={vi.fn()} />);

        const input = screen.getByLabelText(/Upload media files/);

        expect(input).toHaveAttribute(
            'accept',
            '.mp4,.avi,.mkv,.mov,.webm,.m4v,.jpg,.jpeg,.png,.jfif,.tif,.tiff,.webp,.bmp'
        );
    });
});
