// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const formatBytes = (bytes: number) => {
    if (!Number.isFinite(bytes)) {
        return '0 MB';
    }

    const mb = bytes / (1024 * 1024);
    const gb = bytes / (1024 * 1024 * 1024);

    if (gb >= 1) {
        return `${gb.toFixed(2)} GB`;
    }

    return `${mb.toFixed(2)} MB`;
};
