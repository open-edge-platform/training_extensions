// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isMac } from '@react-aria/utils';

import { TaskType } from '../constants/shared-types';

const CTRL_KEY = 'ctrl';
const COMMAND_KEY = 'meta';
const CTRL_OR_COMMAND_KEY = isMac() ? COMMAND_KEY : CTRL_KEY;

export const HOTKEYS = {
    undo: `${CTRL_OR_COMMAND_KEY}+z`,
    redo: `${CTRL_OR_COMMAND_KEY}+y`,
    toggleAnnotationsVisibility: 'a',
    deleteAnnotation: 'delete',
    fitToScreen: 'r',
    selectionTool: 'v',
    boundingBoxTool: 'b',
    autoSegmentation: 's',
    polygonTool: 'p',
    magneticLassoTool: 'm',
    selectAllAnnotations: `${CTRL_OR_COMMAND_KEY}+a`,
    deselectAllAnnotations: `${CTRL_OR_COMMAND_KEY}+d`,
} as const;

const COMMON_HOTKEYS = {
    undo: HOTKEYS.undo,
    redo: HOTKEYS.redo,
    toggleAnnotationsVisibility: HOTKEYS.toggleAnnotationsVisibility,
    deleteAnnotation: HOTKEYS.deleteAnnotation,
    fitToScreen: HOTKEYS.fitToScreen,
    selectAllAnnotations: HOTKEYS.selectAllAnnotations,
    deselectAllAnnotations: HOTKEYS.deselectAllAnnotations,
} as const;

const SELECTION_TOOL_HOTKEY = {
    selectionTool: HOTKEYS.selectionTool,
} as const;

const AUTO_SEGMENTATION_HOTKEY = {
    autoSegmentation: HOTKEYS.autoSegmentation,
} as const;

export const TASK_HOTKEYS = {
    detection: {
        boundingBoxTool: HOTKEYS.boundingBoxTool,
        ...COMMON_HOTKEYS,
        ...SELECTION_TOOL_HOTKEY,
        ...AUTO_SEGMENTATION_HOTKEY,
    },
    instance_segmentation: {
        polygonTool: HOTKEYS.polygonTool,
        magneticLassoTool: HOTKEYS.magneticLassoTool,
        ...COMMON_HOTKEYS,
        ...SELECTION_TOOL_HOTKEY,
        ...AUTO_SEGMENTATION_HOTKEY,
    },
    classification: {
        ...COMMON_HOTKEYS,
    },
} satisfies Record<TaskType, Record<string, string>>;

/**
 * Converts a hotkey string to a user-friendly display format.
 * Replaces 'meta' with 'cmd' and converts to uppercase.
 *
 * @param key - The hotkey string (e.g., 'meta+z' or 'ctrl+z')
 * @returns The formatted hotkey string (e.g., 'CMD+Z' or 'CTRL+Z')
 *
 * @example
 * ```typescript
 * getHotkey('meta+z') // Returns 'CMD+Z' on macOS
 * getHotkey('ctrl+z') // Returns 'CTRL+Z' on Windows/Linux
 * getHotkey('a') // Returns 'A'
 * ```
 */
export const getHotkey = (key: string): string => {
    if (key.includes(COMMAND_KEY)) {
        return key.replace(COMMAND_KEY, 'cmd').toLocaleUpperCase();
    }

    return key.toLocaleUpperCase();
};
