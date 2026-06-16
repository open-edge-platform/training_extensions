// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@geti/ui';

export const downloadFile = (url: string, name?: string, startedMessage?: string): void => {
    const link = document.createElement('a');

    link.href = url;
    if (name !== undefined) {
        link.download = name;
    }
    link.hidden = true;
    link.click();

    if (startedMessage !== undefined) {
        toast({ type: 'info', message: startedMessage });
    }

    if (url.startsWith('blob:')) {
        URL.revokeObjectURL(url);
    }
};
