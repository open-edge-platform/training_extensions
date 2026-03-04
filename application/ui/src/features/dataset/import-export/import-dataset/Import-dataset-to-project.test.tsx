// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { useImportDatasetToProject } from '../../../../hooks/localStorage/use-import-dataset-to-project.hook';
import {
    ImportDatasetDialogStateProvider,
    useImportDatasetDialogState,
} from '../../providers/export-import-dataset-dialog-provider.component';
import { ImportDatasetToProject } from './Import-dataset-to-project.component';

vi.mock('../../../../hooks/localStorage/use-import-dataset-to-project.hook');

describe('ImportDatasetToProject', () => {
    const renderApp = (data: null | { id: string; fileName: string }) => {
        vi.mocked(useImportDatasetToProject).mockReturnValue({
            getImportEntry: vi.fn().mockReturnValue(data),
            getAllImportEntries: vi.fn(),
            appendImportEntry: vi.fn(),
            deleteImportEntry: vi.fn(),
            updateImportEntryStep: vi.fn(),
            updateImportEntry: vi.fn(),
        });

        const App = () => {
            const { datasetImportDialogState } = useImportDatasetDialogState();

            return (
                <>
                    <button onClick={datasetImportDialogState.open}>Open Import Dialog</button>
                    <ImportDatasetToProject />
                </>
            );
        };

        render(
            <ImportDatasetDialogStateProvider>
                <App />
            </ImportDatasetDialogStateProvider>
        );
    };

    it('renders ImportDropZone component in initial state', async () => {
        renderApp(null);

        userEvent.click(screen.getByRole('button', { name: /open import dialog/i }));
        expect(await screen.findByText('Drop the dataset .zip file here')).toBeVisible();
    });
});
