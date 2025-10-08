// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useEventListener } from 'hooks/event-listener.hook';

export const usePanning = () => {
    const [isPanning, setIsPanning] = useState(false);

    useEventListener('keydown', (event: KeyboardEvent) => {
        if (event.code === 'Space') {
            event.preventDefault();
            setIsPanning(true);
        }
    });

    useEventListener('keyup', (event: KeyboardEvent) => {
        if (event.code === 'Space') {
            setIsPanning(false);
        }
    });

    return isPanning;
};
