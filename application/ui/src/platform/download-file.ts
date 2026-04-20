// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { downloadViaAnchor } from './download-file-anchor';

export const downloadFile = (url: string, name?: string): void => {
    downloadViaAnchor(url, name);
};
