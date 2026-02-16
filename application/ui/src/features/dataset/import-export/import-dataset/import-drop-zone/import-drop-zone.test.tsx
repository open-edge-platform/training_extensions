// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { ImportDropZone } from './import-drop-zone.component';

describe('ImportDropZone', () => {
    const validFile = new File(['file content'], 'test.zip', { type: 'application/zip' });
    const inValidFiles = new File(['foo'], 'video.mov', { type: 'video/quicktime' });

    it('invalid file extension', async () => {
        const mockedNextStep = vi.fn();
        render(<ImportDropZone onNextStep={mockedNextStep} />);

        const uploadFileElement = screen.getByTestId(/upload-zip-file/i);

        await userEvent.upload(uploadFileElement, [inValidFiles]);

        expect(screen.getByText(/Unsupported file format. Please upload a valid .zip file./i)).toBeVisible();

        await waitFor(() => {
            expect(mockedNextStep).not.toHaveBeenCalled();
        });
    });

    it('valid file extension', async () => {
        const mockedNextStep = vi.fn();
        render(<ImportDropZone onNextStep={mockedNextStep} />);

        const uploadFileElement = screen.getByTestId(/upload-zip-file/i);

        await userEvent.upload(uploadFileElement, [validFile]);

        await waitFor(() => {
            expect(mockedNextStep).toHaveBeenCalled();
        });
    });
});
