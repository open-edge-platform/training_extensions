// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isMac } from '@react-aria/utils';

const CTRL_KEY = 'ctrl';
export const COMMAND_KEY = 'meta';
const CTRL_OR_COMMAND_KEY = isMac() ? COMMAND_KEY : CTRL_KEY;

type HotKeyActions = 'undo' | 'redo' | 'toggleAnnotations' | 'deleteAnnotation' | 'fitToScreen';

type Hotkeys = Record<HotKeyActions, string>;

export const HOTKEYS: Hotkeys = {
    undo: `${CTRL_OR_COMMAND_KEY}+z`,
    redo: `${CTRL_OR_COMMAND_KEY}+y`,
    toggleAnnotations: 'a',
    deleteAnnotation: 'delete',
    fitToScreen: 'r',
};
