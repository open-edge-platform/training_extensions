// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from 'test-utils/render';

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
});
