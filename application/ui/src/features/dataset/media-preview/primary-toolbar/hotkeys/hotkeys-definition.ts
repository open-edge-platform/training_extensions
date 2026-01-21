// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isMac } from '@react-aria/utils';

const CTRL_KEY = 'ctrl';
export const COMMAND_KEY = 'meta';
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
