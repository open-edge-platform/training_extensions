// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { fireEvent, render, screen } from '@testing-library/react';

import { UndoRedoProvider } from './undo-redo-provider.component';
import useUndoRedoState, { SetStateWrapper } from './use-undo-redo-state';

describe('useUndoRedoState', (): void => {
    const App = ({ forced = false }) => {
        const [value, setValue, undoRedo] = useUndoRedoState(0);

        const increase = () => setValue((oldValue) => oldValue + 1);
        const decrease = () => setValue((oldValue) => oldValue - 1);
        const canUndo = forced || undoRedo.canUndo;
        const canRedo = forced || undoRedo.canRedo;

        return (
            <div>
                <span>{value}</span>
                <button onClick={increase}>Increase</button>
                <button onClick={decrease}>Decrease</button>
                <button onClick={() => undoRedo.undo()} disabled={!canUndo}>
                    Undo
                </button>
                <button onClick={() => undoRedo.redo()} disabled={!canRedo}>
                    Redo
                </button>
                <button onClick={() => undoRedo.reset()} disabled={!canUndo && !canRedo}>
                    Reset
                </button>
            </div>
        );
    };

    const MockAppMultipleCalls = ({ method }: { method: 'undo' | 'redo' }) => {
        const [value, setValue, undoRedo] = useUndoRedoState(0);
        const auxMethod = method === 'undo' ? 'redo' : 'undo';
        useEffect(() => {
            setValue(1);
            // eslint-disable-next-line react-hooks/exhaustive-deps
        }, []);
        return (
            <div>
                <span>{value}</span>
                <button
                    onClick={() => {
                        undoRedo[method]();
                        undoRedo[method]();
                        undoRedo[method]();
                    }}
                >
                    {method}
                </button>
                <button
                    onClick={() => {
                        undoRedo[auxMethod]();
                    }}
                >
                    {auxMethod}
                </button>
            </div>
        );
    };

    it('Can be used like useState', (): void => {
        render(<App />);

        expect(screen.getByText('0')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Increase' }));
        fireEvent.click(screen.getByRole('button', { name: 'Increase' }));
        expect(screen.getByText('2')).toBeInTheDocument();
    });

    it('Can undo a set state action', (): void => {
        render(<App />);

        const undoButton = screen.getByRole('button', { name: 'Undo' });
        expect(undoButton).toBeDisabled();

        expect(screen.getByText('0')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Increase' }));
        expect(screen.getByText('1')).toBeInTheDocument();
        expect(undoButton).not.toBeDisabled();

        fireEvent.click(undoButton);
        expect(screen.getByText('0')).toBeInTheDocument();
        expect(undoButton).toBeDisabled();
    });

    it('Can redo a undo', (): void => {
        render(<App />);

        const undoButton = screen.getByRole('button', { name: 'Undo' });
        expect(undoButton).toBeDisabled();

        const redoButton = screen.getByRole('button', { name: 'Redo' });
        expect(redoButton).toBeDisabled();

        expect(screen.getByText('0')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Increase' }));
        expect(screen.getByText('1')).toBeInTheDocument();
        expect(undoButton).not.toBeDisabled();
        expect(redoButton).toBeDisabled();

        fireEvent.click(undoButton);
        expect(screen.getByText('0')).toBeInTheDocument();
        expect(undoButton).toBeDisabled();
        expect(redoButton).not.toBeDisabled();
    });

    it('Can reset to the initial state', (): void => {
        render(<App />);

        const resetButton = screen.getByRole('button', { name: 'Reset' });
        expect(resetButton).toBeDisabled();

        expect(screen.getByText('0')).toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Increase' }));
        expect(screen.getByText('1')).toBeInTheDocument();
        expect(resetButton).not.toBeDisabled();

        fireEvent.click(resetButton);
        expect(screen.getByText('0')).toBeInTheDocument();
    });

    it('Can reset to the initial state to a value', (): void => {
        const MockApp = () => {
            const [value, , undoRedo] = useUndoRedoState(0);

            return (
                <div>
                    <span>{value}</span>
                    <button onClick={() => undoRedo.reset(33)}>Reset</button>
                </div>
            );
        };
        render(<MockApp />);

        const resetButton = screen.getByRole('button', { name: 'Reset' });
        fireEvent.click(resetButton);
        expect(screen.getByText('33')).toBeInTheDocument();
    });

    it('Ignores subsequent calls to undo', (): void => {
        render(<App forced />);

        const undoButton = screen.getByRole('button', { name: 'Undo' });
        fireEvent.click(undoButton);
        expect(screen.getByText('0')).toBeInTheDocument();

        const redoButton = screen.getByRole('button', { name: 'Redo' });
        fireEvent.click(redoButton);
        expect(screen.getByText('0')).toBeInTheDocument();

        const resetButton = screen.getByRole('button', { name: 'Reset' });
        fireEvent.click(resetButton);
        expect(screen.getByText('0')).toBeInTheDocument();
    });

    it('Overwrites the history stack after undoing an action', (): void => {
        render(<App />);

        const undoButton = screen.getByRole('button', { name: 'Undo' });
        const increaseButton = screen.getByRole('button', { name: 'Increase' });
        const decreaseButton = screen.getByRole('button', { name: 'Decrease' });
        fireEvent.click(increaseButton);
        fireEvent.click(decreaseButton);
        fireEvent.click(undoButton);
        fireEvent.click(increaseButton);
        expect(screen.getByText('2')).toBeInTheDocument();

        const redoButton = screen.getByRole('button', { name: 'Redo' });
        expect(redoButton).toBeDisabled();
    });

    it('Overwrites the history stack after resetting', (): void => {
        render(<App />);

        const increaseButton = screen.getByRole('button', { name: 'Increase' });
        const resetButton = screen.getByRole('button', { name: 'Reset' });
        fireEvent.click(increaseButton);
        fireEvent.click(resetButton);
        fireEvent.click(increaseButton);
        expect(screen.getByText('1')).toBeInTheDocument();
    });

    it('Allows skipping the undo redo history', () => {
        const MockApp = () => {
            const [value, setValue, undoRedo] = useUndoRedoState(0);
            const increase = () => setValue((oldValue) => oldValue + 1, true);

            return (
                <div>
                    <span>{value}</span>
                    <button onClick={increase}>Increase</button>
                    <button onClick={undoRedo.undo} disabled={!undoRedo.canUndo}>
                        Undo
                    </button>
                </div>
            );
        };

        render(<MockApp />);

        fireEvent.click(screen.getByRole('button', { name: 'Increase' }));

        const undoButton = screen.getByRole('button', { name: 'Undo' });
        expect(undoButton).toBeDisabled();
        expect(screen.getByText('1')).toBeInTheDocument();
    });

    it("Won't fail when calling setState multiple times in one render", () => {
        const MockApp = () => {
            const [value, setValue, undoRedo] = useUndoRedoState(0);
            const increase = () => {
                setValue(1);
                setValue(2);
                setValue(3);
            };

            return (
                <div>
                    <span>{value}</span>
                    <button onClick={increase}>Increase</button>
                    <button onClick={undoRedo.undo} disabled={!undoRedo.canUndo}>
                        Undo
                    </button>
                </div>
            );
        };

        render(<MockApp />);

        fireEvent.click(screen.getByRole('button', { name: 'Increase' }));

        expect(screen.getByText('3')).toBeInTheDocument();
        const undoButton = screen.getByRole('button', { name: 'Undo' });

        fireEvent.click(undoButton);
        expect(screen.getByText('2')).toBeInTheDocument();

        fireEvent.click(undoButton);
        expect(screen.getByText('1')).toBeInTheDocument();
    });

    it("Won't fail when calling undo multiple times", (): void => {
        render(<MockAppMultipleCalls method={'undo'} />);

        const undoButton = screen.getByRole('button', { name: 'undo' });
        fireEvent.click(undoButton);
        expect(screen.getByText('0')).toBeInTheDocument();
    });

    it("Won't fail when calling redo multiple times", (): void => {
        render(<MockAppMultipleCalls method={'redo'} />);

        const undoButton = screen.getByRole('button', { name: 'undo' });
        fireEvent.click(undoButton);
        expect(screen.getByText('0')).toBeInTheDocument();

        const redoButton = screen.getByRole('button', { name: 'redo' });
        fireEvent.click(redoButton);
        expect(screen.getByText('1')).toBeInTheDocument();
    });

    describe('Undo redo from a parent undo redo state', () => {
        const MockApp = ({
            parentValue,
            setParentValue,
        }: {
            parentValue: number;
            setParentValue: SetStateWrapper<number>;
        }) => {
            const [value, setValue, undoRedo] = useUndoRedoState(0);
            const increase = () => setValue((oldValue) => oldValue + 1);
            const increaseParent = () => setParentValue((oldValue) => oldValue + 1);

            return (
                <div>
                    <span aria-label='Child value'>{value}</span>
                    <span aria-label='Parent value'>{parentValue}</span>
                    <button onClick={increase}>Increase</button>
                    <button onClick={increaseParent}>Increase parent</button>
                    <button onClick={undoRedo.undo} disabled={!undoRedo.canUndo}>
                        Undo
                    </button>
                    <button onClick={undoRedo.redo} disabled={!undoRedo.canRedo}>
                        Redo
                    </button>
                </div>
            );
        };

        const ParentApp = () => {
            const [value, setValue, undoRedo] = useUndoRedoState(0);

            return (
                <UndoRedoProvider state={undoRedo}>
                    <MockApp parentValue={value} setParentValue={setValue} />
                </UndoRedoProvider>
            );
        };

        it("Undoes a parent's undo redo changes", () => {
            render(<ParentApp />);

            const increaseButton = screen.getByRole('button', { name: 'Increase' });
            const increaseParentButton = screen.getByRole('button', { name: 'Increase parent' });

            fireEvent.click(increaseParentButton);

            const undoButton = screen.getByRole('button', { name: 'Undo' });
            const redoButton = screen.getByRole('button', { name: 'Redo' });
            expect(undoButton).not.toBeDisabled();

            fireEvent.click(increaseParentButton);
            fireEvent.click(increaseButton);

            expect(screen.getByLabelText('Child value')).toHaveTextContent('1');
            expect(screen.getByLabelText('Parent value')).toHaveTextContent('2');

            fireEvent.click(undoButton);
            expect(screen.getByLabelText('Child value')).toHaveTextContent('0');
            expect(screen.getByLabelText('Parent value')).toHaveTextContent('2');

            fireEvent.click(redoButton);
            expect(screen.getByLabelText('Child value')).toHaveTextContent('1');
            expect(screen.getByLabelText('Parent value')).toHaveTextContent('2');

            fireEvent.click(undoButton);
            expect(screen.getByLabelText('Child value')).toHaveTextContent('0');
            expect(screen.getByLabelText('Parent value')).toHaveTextContent('2');

            fireEvent.click(undoButton);
            expect(screen.getByLabelText('Child value')).toHaveTextContent('0');
            expect(screen.getByLabelText('Parent value')).toHaveTextContent('1');

            // Parent should be redoed
            fireEvent.click(redoButton);
            expect(screen.getByLabelText('Child value')).toHaveTextContent('0');
            expect(screen.getByLabelText('Parent value')).toHaveTextContent('2');

            // Next the child should be reoded
            fireEvent.click(redoButton);
            expect(screen.getByLabelText('Child value')).toHaveTextContent('1');
            expect(screen.getByLabelText('Parent value')).toHaveTextContent('2');
            expect(redoButton).toBeDisabled();
        });
    });
});
