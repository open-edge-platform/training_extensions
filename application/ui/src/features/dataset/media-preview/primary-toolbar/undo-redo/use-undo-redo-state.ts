// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SetStateAction, useCallback, useState } from 'react';

import { UndoRedoActions } from './undo-redo-actions.interface';
import { useParentUndoRedo } from './undo-redo-provider.component';

export interface SetStateWrapper<State> {
    (value: SetStateAction<State>, skipHistory?: boolean): void;
}

function getInitialValue<State>(newState: State | (() => State)) {
    if (!(typeof newState === 'function')) {
        return newState;
    }

    // Use lazy initialization
    const initializeState = newState as () => State;

    return initializeState();
}

function getNewValue<State>(newState: SetStateAction<State>, value: State) {
    if (!(typeof newState === 'function')) {
        return newState;
    }

    // If the user provided a function we will want to compute the new state based
    // on the old state
    const computeState = newState as (prevState: State) => State;

    return computeState(value);
}

type UseUndoRedoState<State> = [State, SetStateWrapper<State>, UndoRedoActions<State>];

function useUndoRedoState<State>(initialState: State | (() => State)): UseUndoRedoState<State> {
    const parentUndoRedo = useParentUndoRedo();

    const [state, setState] = useState(() => ({
        state: getInitialValue(initialState),
        history: [getInitialValue(initialState)],
        index: 0,
    }));

    // Undoing current state has precedence over undoing parent state
    const canUndoParent = parentUndoRedo !== undefined && parentUndoRedo.canUndo;
    const canUndo = state.index > 0 || canUndoParent;

    const undo = useCallback(() => {
        if (!canUndo) {
            return;
        }

        if (state.index > 0) {
            setState((prevState) => {
                const { history, index } = prevState;
                const newState = { state: history[index - 1], history, index: index - 1 };

                return newState.index < 0 ? prevState : newState;
            });
        } else {
            parentUndoRedo?.undo();
        }
    }, [canUndo, state, parentUndoRedo]);

    // Redoing parent state has precedence over undoing current state
    const canRedoParent = parentUndoRedo !== undefined && parentUndoRedo.canRedo;
    const canRedo = state.index < state.history.length - 1 || canRedoParent;

    const redo = useCallback(() => {
        if (!canRedo) {
            return;
        }

        if (parentUndoRedo?.canRedo) {
            parentUndoRedo.redo();
        } else {
            setState((prevState) => {
                const { history, index } = prevState;
                const newState = { state: history[index + 1], history, index: index + 1 };

                return index < history.length - 1 ? newState : prevState;
            });
        }
    }, [canRedo, parentUndoRedo]);

    const reset = useCallback((optionalResetState?: State) => {
        setState(({ history }) => {
            const resetState = optionalResetState === undefined ? history[0] : optionalResetState;

            return { state: resetState, index: 0, history: [resetState] };
        });
    }, []);

    const undoRedoActions = { canUndo, canRedo, undo, redo, reset };

    const setStateWrapper: SetStateWrapper<State> = (newState, skipHistory = false) => {
        setState(({ state: oldState, history, index }) => {
            const actualNewState = getNewValue(newState, oldState);

            if (skipHistory) {
                return { state: actualNewState, history, index };
            }

            const newIndex = index + 1;

            return {
                state: actualNewState,
                history: [...history.slice(0, newIndex), actualNewState],
                index: newIndex,
            };
        });
    };

    return [state.state, setStateWrapper, undoRedoActions];
}

export default useUndoRedoState;
