// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useHotkeys } from 'react-hotkeys-hook';

export const usePanning = () => {
    const [isPanning, setIsPanning] = useState(false);

    useHotkeys('space', () => setIsPanning(true), { keydown: true, enableOnContentEditable: true });

    useHotkeys('space', () => setIsPanning(false), { keyup: true, enableOnContentEditable: true });

    return { isPanning, setIsPanning };
};
