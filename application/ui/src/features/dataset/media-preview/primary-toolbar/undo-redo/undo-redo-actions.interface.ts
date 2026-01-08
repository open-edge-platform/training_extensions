// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export interface UndoRedoActions<State = unknown> {
    readonly canUndo: boolean;
    readonly canRedo: boolean;

    undo(): void;
    redo(): void;
    reset(state?: State): void;
}
