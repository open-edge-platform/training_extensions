// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import React from 'react';

import ReactDOM from 'react-dom/client';

import { Providers } from './providers';
import { setupTauriStorageCleanup } from './tauri/utils/storage-cleanup-registry/storage-cleanup-registry';

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
