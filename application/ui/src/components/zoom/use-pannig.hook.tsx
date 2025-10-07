// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useEventListener } from 'hooks/event-listener.hook';

import { KeyboardEvents, KeyMap } from '../../constants/keyboard.interface';

export const usePanning = () => {
    const [isPanning, setIsPanning] = useState(false);

    useEventListener(KeyboardEvents.KeyDown, (event: KeyboardEvent) => {
        if (event.code === KeyMap.Space) {
            event.preventDefault();
            setIsPanning(true);
        }
    });

    useEventListener(KeyboardEvents.KeyUp, (event: KeyboardEvent) => {
        if (event.code === KeyMap.Space) {
            setIsPanning(false);
        }
    });

    return isPanning;
};
