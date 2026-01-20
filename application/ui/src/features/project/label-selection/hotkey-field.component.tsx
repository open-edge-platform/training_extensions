// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { TextField } from '@geti/ui';

type HotkeyFieldProps = {
    hotkey: string | null | undefined;
    onHotkeyChange: (hotkey: string | null) => void;
};

export const HotkeyField = ({ hotkey, onHotkeyChange }: HotkeyFieldProps) => {
    return (
        <TextField
            aria-label={'Hotkey input'}
            placeholder={'Hotkey'}
            value={hotkey ?? undefined}
            onChange={onHotkeyChange}
            width={'100%'}
        />
    );
};
