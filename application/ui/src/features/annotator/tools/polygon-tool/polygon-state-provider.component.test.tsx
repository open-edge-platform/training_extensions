// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import {
    UndoRedoProvider,
    useUndoRedo,
} from '../../../dataset/media-preview/primary-toolbar/undo-redo/undo-redo-provider.component';
import useUndoRedoState, {
    SetStateWrapper,
} from '../../../dataset/media-preview/primary-toolbar/undo-redo/use-undo-redo-state';
import type { ToolType } from '../interface';
import { PolygonStateProvider, usePolygonState } from './polygon-state-provider.component';

const mockSetActiveTool = vi.fn();
let mockActiveTool: ToolType | null = 'polygon';

vi.mock('../../../../shared/annotator/tool-provider.component', () => ({
    useTool: () => ({ activeTool: mockActiveTool, setActiveTool: mockSetActiveTool }),
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * A consumer component that exposes the polygon drawing state via data-testid
 * attributes so tests can assert on the current values.
 */
const PolygonStateConsumer = () => {
    const { segments, pointerLine, lassoSegment, setSegments, setPointerLine, setLassoSegment, undoRedoActions } =
        usePolygonState();

    return (
        <div>
            <span data-testid='segments'>{JSON.stringify(segments)}</span>
            <span data-testid='pointer-line'>{JSON.stringify(pointerLine)}</span>
            <span data-testid='lasso-segment'>{JSON.stringify(lassoSegment)}</span>

            <button onClick={() => setSegments((prev) => [...prev, [{ x: 1, y: 1 }]])}>Add segment</button>
            <button onClick={() => setPointerLine([{ x: 2, y: 2 }])}>Set pointer line</button>
            <button onClick={() => setLassoSegment([{ x: 3, y: 3 }])}>Set lasso segment</button>

            <button onClick={undoRedoActions.undo} disabled={!undoRedoActions.canUndo}>
                Undo
            </button>
            <button onClick={undoRedoActions.redo} disabled={!undoRedoActions.canRedo}>
                Redo
            </button>
        </div>
    );
};

/**
 * A component that reads the UndoRedo context (resolves to the innermost
 * UndoRedoProvider) so tests can assert toolbar-level undo/redo behaviour.
 */
const ToolbarUndoRedoConsumer = () => {
    const { undo, redo, canUndo, canRedo } = useUndoRedo();

    return (
        <div>
            <button onClick={undo} disabled={!canUndo}>
                Toolbar undo
            </button>
            <button onClick={redo} disabled={!canRedo}>
                Toolbar redo
            </button>
        </div>
    );
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('PolygonStateProvider', () => {
    beforeEach(() => {
        mockActiveTool = 'polygon';
    });

    const renderProvider = () =>
        render(
            <PolygonStateProvider>
                <PolygonStateConsumer />
                <ToolbarUndoRedoConsumer />
            </PolygonStateProvider>
        );

    it('provides initial empty drawing state', () => {
        renderProvider();

        expect(screen.getByTestId('segments')).toHaveTextContent('[]');
        expect(screen.getByTestId('pointer-line')).toHaveTextContent('[]');
        expect(screen.getByTestId('lasso-segment')).toHaveTextContent('[]');
    });

    it('toolbar undo button is disabled when no segments have been added', () => {
        renderProvider();

        expect(screen.getByRole('button', { name: 'Toolbar undo' })).toBeDisabled();
    });

    it('toolbar undo removes the last segment', () => {
        renderProvider();

        fireEvent.click(screen.getByRole('button', { name: 'Add segment' }));
        fireEvent.click(screen.getByRole('button', { name: 'Add segment' }));
        expect(JSON.parse(screen.getByTestId('segments').textContent ?? '[]')).toHaveLength(2);

        fireEvent.click(screen.getByRole('button', { name: 'Toolbar undo' }));
        expect(JSON.parse(screen.getByTestId('segments').textContent ?? '[]')).toHaveLength(1);
    });

    it('toolbar undo clears pointerLine and lassoSegment (fixes frozen-tool bug)', () => {
        renderProvider();

        fireEvent.click(screen.getByRole('button', { name: 'Add segment' }));
        fireEvent.click(screen.getByRole('button', { name: 'Set pointer line' }));
        fireEvent.click(screen.getByRole('button', { name: 'Set lasso segment' }));

        expect(screen.getByTestId('pointer-line')).not.toHaveTextContent('[]');
        expect(screen.getByTestId('lasso-segment')).not.toHaveTextContent('[]');

        fireEvent.click(screen.getByRole('button', { name: 'Toolbar undo' }));

        expect(screen.getByTestId('pointer-line')).toHaveTextContent('[]');
        expect(screen.getByTestId('lasso-segment')).toHaveTextContent('[]');
    });

    it('toolbar redo clears pointerLine and lassoSegment', () => {
        renderProvider();

        fireEvent.click(screen.getByRole('button', { name: 'Add segment' }));
        fireEvent.click(screen.getByRole('button', { name: 'Toolbar undo' }));

        // Set transient state after undo to simulate pointer activity
        fireEvent.click(screen.getByRole('button', { name: 'Set pointer line' }));
        fireEvent.click(screen.getByRole('button', { name: 'Set lasso segment' }));

        fireEvent.click(screen.getByRole('button', { name: 'Toolbar redo' }));

        expect(screen.getByTestId('pointer-line')).toHaveTextContent('[]');
        expect(screen.getByTestId('lasso-segment')).toHaveTextContent('[]');
    });

    it('resets all drawing state when activeTool changes', () => {
        const tree = (
            <PolygonStateProvider>
                <PolygonStateConsumer />
                <ToolbarUndoRedoConsumer />
            </PolygonStateProvider>
        );
        const { rerender } = render(tree);

        fireEvent.click(screen.getByRole('button', { name: 'Add segment' }));
        fireEvent.click(screen.getByRole('button', { name: 'Set pointer line' }));
        fireEvent.click(screen.getByRole('button', { name: 'Set lasso segment' }));

        expect(JSON.parse(screen.getByTestId('segments').textContent ?? '[]')).toHaveLength(1);
        expect(screen.getByTestId('pointer-line')).not.toHaveTextContent('[]');
        expect(screen.getByTestId('lasso-segment')).not.toHaveTextContent('[]');

        // Simulate user switching to a different tool
        mockActiveTool = 'bounding-box';
        rerender(tree);

        expect(screen.getByTestId('segments')).toHaveTextContent('[]');
        expect(screen.getByTestId('pointer-line')).toHaveTextContent('[]');
        expect(screen.getByTestId('lasso-segment')).toHaveTextContent('[]');
    });

    describe('undo/redo chaining with parent annotations stack', () => {
        /**
         * Sets up the full provider chain:
         *   ParentAnnotationsStack (UndoRedoProvider)
         *     └─ PolygonStateProvider (inner UndoRedoProvider)
         *          └─ PolygonStateConsumer + ToolbarUndoRedoConsumer
         *
         * The ToolbarUndoRedoConsumer reads from the innermost UndoRedoProvider,
         * which is PolygonStateProvider's. When the polygon stack is empty the
         * existing chaining in useUndoRedoState delegates to the parent.
         */
        const ParentAnnotationsApp = ({
            parentValue,
            setParentValue,
        }: {
            parentValue: number;
            setParentValue: SetStateWrapper<number>;
        }) => (
            <div>
                <span data-testid='parent-value'>{parentValue}</span>
                <button onClick={() => setParentValue((v) => v + 1)}>Increment parent</button>
                <PolygonStateProvider>
                    <PolygonStateConsumer />
                    <ToolbarUndoRedoConsumer />
                </PolygonStateProvider>
            </div>
        );

        const renderWithParent = () => {
            const ParentWrapper = () => {
                const [value, setValue, undoRedo] = useUndoRedoState(0);

                return (
                    <UndoRedoProvider state={undoRedo}>
                        <ParentAnnotationsApp parentValue={value} setParentValue={setValue} />
                    </UndoRedoProvider>
                );
            };

            return render(<ParentWrapper />);
        };

        it('delegates to the parent (annotations) stack once the polygon stack is empty', () => {
            renderWithParent();

            // Commit a change to the parent (annotations) stack
            fireEvent.click(screen.getByRole('button', { name: 'Increment parent' }));
            expect(screen.getByTestId('parent-value')).toHaveTextContent('1');

            // Add a polygon segment (polygon stack depth = 1)
            fireEvent.click(screen.getByRole('button', { name: 'Add segment' }));

            // First toolbar undo should pop the polygon segment
            fireEvent.click(screen.getByRole('button', { name: 'Toolbar undo' }));
            expect(JSON.parse(screen.getByTestId('segments').textContent ?? '[]')).toHaveLength(0);
            expect(screen.getByTestId('parent-value')).toHaveTextContent('1');

            // Second toolbar undo: polygon stack empty → delegates to parent
            fireEvent.click(screen.getByRole('button', { name: 'Toolbar undo' }));
            expect(screen.getByTestId('parent-value')).toHaveTextContent('0');
        });
    });
});
