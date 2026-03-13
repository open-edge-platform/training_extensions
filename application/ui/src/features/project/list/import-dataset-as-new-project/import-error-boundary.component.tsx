// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ErrorBoundary } from 'react-error-boundary';

import { useImportDatasetDialog } from '../../providers/import-dataset-dialog-provider.component';

export const ImportErrorBoundary = ({ children }: { children: React.ReactNode }) => {
    const { datasetImportDialogState } = useImportDatasetDialog();

    return (
        <ErrorBoundary
            FallbackComponent={() => {
                datasetImportDialogState.close();
                return null;
            }}
        >
            {children}
        </ErrorBoundary>
    );
};
