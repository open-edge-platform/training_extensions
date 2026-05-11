// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export type Action =
    | { type: HOTKEY_EDITION_ACTION.ADD_KEY; keys: string[] }
    | { type: HOTKEY_EDITION_ACTION.CHANGE_FOCUS; isFocused: boolean };

export enum HOTKEY_EDITION_ACTION {
    CHANGE_FOCUS,
    ADD_KEY,
}
