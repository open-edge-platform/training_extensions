// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const downloadViaAnchor = (url: string, name?: string): void => {
    const link = document.createElement('a');

    link.href = url;
    if (name !== undefined) {
        link.download = name;
    }
    link.hidden = true;
    link.click();

    if (url.startsWith('blob:')) {
        URL.revokeObjectURL(url);
    }
};
