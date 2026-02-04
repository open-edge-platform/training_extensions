// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label } from '../../constants/shared-types';
import { convertHotkeyToOSFormat } from '../../shared/hotkeys-definition';

export const validateLabelName = (
    name: string,
    existingLabels: Label[],
    excludeId?: string
): string | undefined => {
    const trimmedName = name.trim();

    const isDuplicate = existingLabels.some((label) => label.name === trimmedName && label.id !== excludeId);

    if (isDuplicate) {
        return 'That label name already exists';
    }

    return undefined;
};

export const validateLabelHotkey = (hotkey: string, allHotkeys: string[]): string | undefined => {
    const osFormatHotkeys = allHotkeys.map(convertHotkeyToOSFormat).map((key) => key.toLowerCase());

    if (osFormatHotkeys.includes(hotkey.toLowerCase())) {
        return 'That hotkey is already in use';
    }

    return undefined;
};
