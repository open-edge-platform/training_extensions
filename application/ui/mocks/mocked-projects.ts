// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaProjectView } from '../src/api/openapi-spec';

export const mockedProjects: SchemaProjectView[] = [
    {
        id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
        name: 'Production Pipeline',
        task: {
            task_type: 'detection',
            exclusive_labels: false,
            labels: [
                { id: '1', name: 'Box', color: '#FF0000', hotkey: '1' },
                { id: '2', name: 'Paper', color: '#00FF00', hotkey: '2' },
                { id: '3', name: 'Blocks', color: '#0000FF', hotkey: '3' },
            ],
        },
        active_pipeline: true,
    },
    {
        id: 'e2f3g4h5-i6j7-8901-bcde-fg2345678901',
        name: 'Testing Pipeline',
        task: {
            task_type: 'detection',
            exclusive_labels: false,
            labels: [
                { id: '1', name: 'Box', color: '#FF0000', hotkey: '1' },
                { id: '2', name: 'Paper', color: '#00FF00', hotkey: '2' },
                { id: '3', name: 'Blocks', color: '#0000FF', hotkey: '3' },
            ],
        },
        active_pipeline: false,
    },
    {
        id: 'i3j4k5l6-m7n8-9012-cdef-hi3456789012',
        name: 'Development Pipeline',
        task: {
            task_type: 'detection',
            exclusive_labels: false,
            labels: [
                { id: '1', name: 'Box', color: '#FF0000', hotkey: '1' },
                { id: '2', name: 'Paper', color: '#00FF00', hotkey: '2' },
                { id: '3', name: 'Blocks', color: '#0000FF', hotkey: '3' },
            ],
        },
        active_pipeline: false,
    },
    {
        id: 'm4n5o6p7-q8r9-0123-defg-jk4567890123',
        name: 'Analytics Pipeline',
        task: {
            task_type: 'detection',
            exclusive_labels: false,
            labels: [
                { id: '1', name: 'Box', color: '#FF0000', hotkey: '1' },
                { id: '2', name: 'Paper', color: '#00FF00', hotkey: '2' },
                { id: '3', name: 'Blocks', color: '#0000FF', hotkey: '3' },
            ],
        },
        active_pipeline: false,
    },
    {
        id: 's5t6u7v8-w9x0-1234-efgh-lm5678901234',
        name: 'Backup Pipeline',
        task: {
            task_type: 'detection',
            exclusive_labels: false,
            labels: [
                { id: '1', name: 'Box', color: '#FF0000', hotkey: '1' },
                { id: '2', name: 'Paper', color: '#00FF00', hotkey: '2' },
                { id: '3', name: 'Blocks', color: '#0000FF', hotkey: '3' },
            ],
        },
        active_pipeline: false,
    },
];
