// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@test-utils/render';
import userEvent from '@testing-library/user-event';

import { AddMediaButton } from './add-media-button.component';

describe('AddMediaButton', () => {
    it('calls onFilesSelected correctly', async () => {
        const mockOnFilesSelected = vi.fn();
        const mockFile = new File(['file content'], 'test-image.jpg', {
            type: 'image/jpeg',
            lastModified: Date.now(),
        });

        render(<AddMediaButton onFilesSelected={mockOnFilesSelected} />);

        const input = screen.getByLabelText(/Upload media files/);
        await userEvent.upload(input, mockFile);

        expect(mockOnFilesSelected).toHaveBeenCalledWith([mockFile]);
    });
});
