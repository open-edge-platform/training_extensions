// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    createContext,
    Dispatch,
    PropsWithChildren,
    SetStateAction,
    useCallback,
    useContext,
    useMemo,
    useRef,
    useState,
} from 'react';

import { useTool } from '../../../../shared/annotator/tool-provider.component';
import { Point } from '../../../../shared/types';
import { UndoRedoActions } from '../../../dataset/media-preview/primary-toolbar/undo-redo/undo-redo-actions.interface';
import { UndoRedoProvider } from '../../../dataset/media-preview/primary-toolbar/undo-redo/undo-redo-provider.component';
import useUndoRedoState, {
    SetStateWrapper,
} from '../../../dataset/media-preview/primary-toolbar/undo-redo/use-undo-redo-state';

type PolygonState = {
    segments: Point[][];
    setSegments: SetStateWrapper<Point[][]>;
    pointerLine: Point[];
    setPointerLine: Dispatch<SetStateAction<Point[]>>;
    lassoSegment: Point[];
    setLassoSegment: Dispatch<SetStateAction<Point[]>>;
    undoRedoActions: UndoRedoActions<Point[][]>;
};

const PolygonStateContext = createContext<PolygonState | null>(null);

export const PolygonStateProvider = ({ children }: PropsWithChildren) => {
    const { activeTool } = useTool();

    const [pointerLine, setPointerLine] = useState<Point[]>([]);
    const [lassoSegment, setLassoSegment] = useState<Point[]>([]);
    const [segments, setSegments, undoRedoActions] = useUndoRedoState<Point[][]>([]);

    const { canUndo, canRedo, undo, redo, reset } = undoRedoActions;

    const prevToolRef = useRef(activeTool);

    if (prevToolRef.current !== activeTool) {
        prevToolRef.current = activeTool;
        setPointerLine([]);
        setLassoSegment([]);
        reset();
    }

    // Wrap undo/redo so that transient pointer/lasso visuals are cleared
    // immediately when stepping through history. Without this, pointerLine
    // and lassoSegment (which live outside the undo stack) would keep
    // displaying a stale follow-line until the next pointer-move event,
    // making the tool look frozen after an undo.
    const enhancedUndo = useCallback(() => {
        setPointerLine([]);
        setLassoSegment([]);
        undo();
    }, [undo]);

    const enhancedRedo = useCallback(() => {
        setPointerLine([]);
        setLassoSegment([]);
        redo();
    }, [redo]);

    const wrappedUndoRedoActions = useMemo<UndoRedoActions<Point[][]>>(
        () => ({ canUndo, canRedo, undo: enhancedUndo, redo: enhancedRedo, reset }),
        [canUndo, canRedo, enhancedUndo, enhancedRedo, reset]
    );

    const value = useMemo<PolygonState>(
        () => ({ segments, setSegments, pointerLine, setPointerLine, lassoSegment, setLassoSegment, undoRedoActions }),
        [segments, setSegments, pointerLine, lassoSegment, undoRedoActions]
    );

    return (
        <PolygonStateContext.Provider value={value}>
            <UndoRedoProvider state={wrappedUndoRedoActions}>{children}</UndoRedoProvider>
        </PolygonStateContext.Provider>
    );
};

export const usePolygonState = (): PolygonState => {
    const context = useContext(PolygonStateContext);

    if (context === null) {
        throw new Error('usePolygonState must be used within a PolygonStateProvider');
    }

    return context;
};
