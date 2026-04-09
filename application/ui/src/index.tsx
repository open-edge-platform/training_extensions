// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import React from 'react';

import ReactDOM from 'react-dom/client';

import { setupTauriStorageCleanup } from './hooks/storage/storage-cleanup-registry';
import { Providers } from './providers';

import './index.css';

setupTauriStorageCleanup();

const rootEl = document.getElementById('root');
if (rootEl) {
    const root = ReactDOM.createRoot(rootEl);
    root.render(
        <React.StrictMode>
            <Providers />
        </React.StrictMode>
    );
}
