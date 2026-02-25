// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { usePrepareImportDataset } from '../../../../hooks/localStorage/use-prepare-import-dataset.hook';
import { ImportDataset } from './import-dataset.component';

vi.mock('../../../../hooks/localStorage/use-prepare-import-dataset.hook');

const mockDialogState = {
    isOpen: true,
    open: vi.fn(),
    close: vi.fn(),
    toggle: vi.fn(),
    setOpen: vi.fn(),
};

describe('ImportDataset', () => {
    const renderApp = (data: null | { id: string; fileName: string }) => {
        vi.mocked(usePrepareImportDataset).mockReturnValue({
            getLsPreparingImport: vi.fn().mockReturnValue(data),
            addLsPreparingImport: vi.fn(),
            removeLsPreparingImport: vi.fn(),
        });

        render(<ImportDataset dialogState={{ ...mockDialogState, isOpen: true }} />);
    };

    it('renders ImportDropZone component in initial state', () => {
        renderApp(null);

        expect(screen.getByText('Drop the dataset .zip file here')).toBeVisible();
    });

    it('renders ImportProcess component when there is a preparing import in localStorage', () => {
        renderApp({ id: 'test-job-id', fileName: 'test-dataset.zip' });

        expect(screen.getByText('Prepare dataset import to existing project')).toBeVisible();
    });
});
