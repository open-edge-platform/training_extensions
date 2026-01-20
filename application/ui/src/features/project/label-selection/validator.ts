// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Label } from '../../../constants/shared-types';
import { convertHotkeyToOSFormat } from '../../../shared/hotkeys-definition';

export const validateLabelName = (label: Label, labels: Label[]): string | undefined => {
    if (label.name.trim().length === 0) {
        return 'Label name cannot be empty';
    }

    if (labels.some((l) => l.name === label.name)) {
        return 'That label name already exists';
    }

    return undefined;
};

export const validateLabelHotkey = (hotkey: string, allHotkeys: string[]) => {
    const osFormatHotkeys = allHotkeys.map(convertHotkeyToOSFormat).map((key) => key.toLowerCase());

    console.log({ osFormatHotkeys, hotkey });

    if (osFormatHotkeys.includes(hotkey.toLowerCase())) {
        return 'That hotkey is already in use';
    }

    return undefined;
};
