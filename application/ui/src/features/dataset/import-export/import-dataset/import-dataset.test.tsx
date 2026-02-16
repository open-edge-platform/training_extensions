// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { ImportDataset } from './import-dataset.component';

const mockDialogState = {
    isOpen: true,
    open: vi.fn(),
    close: vi.fn(),
    toggle: vi.fn(),
    setOpen: vi.fn(),
};

describe('ImportDataset', () => {
    it('renders ImportDropZone component in initial state', () => {
        render(<ImportDataset dialogState={{ ...mockDialogState, isOpen: true }} />);

        expect(screen.getByText('Drop the dataset .zip file here')).toBeVisible();
    });
});
