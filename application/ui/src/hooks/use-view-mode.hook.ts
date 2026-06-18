// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ViewModes } from '@geti/ui';
import { useLocalStorage } from 'usehooks-ts';

const VIEW_MODE_KEY = 'view-mode';

const getMediaViewModeKey = (suffix: string) => {
    return `${VIEW_MODE_KEY}-${suffix}`;
};

export const useViewMode = (suffix: string, defaultViewMode: ViewModes = ViewModes.MEDIUM) => {
    return useLocalStorage(getMediaViewModeKey(suffix), defaultViewMode);
};
