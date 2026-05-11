// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { getMockedLabel } from '../../../../../mocks/mock-labels';
import { LabelRow, LabelRowProps } from './label-row.component';

describe('LabelRow', () => {
    const renderApp = ({
        label = getMockedLabel(),
        isSelected = false,
        isPinned = false,
        onSelect = vi.fn(),
        onDelete = vi.fn(),
        onTogglePin = vi.fn(),
        onUpdate = vi.fn(),
        validateName = vi.fn(() => undefined),
    }: Partial<LabelRowProps> = {}) => {
        render(
            <LabelRow
                label={label}
                isSelected={isSelected}
                isPinned={isPinned}
                onSelect={onSelect}
                onDelete={onDelete}
                onTogglePin={onTogglePin}
                onUpdate={onUpdate}
                validateName={validateName}
            />
        );

        return { onUpdate };
    };

    describe('handleUpdateName', () => {
        const label = getMockedLabel({ id: 'label-1', name: 'old name', color: '#ffff00', hotkey: 'a' });

        it('calls onUpdate with trimmed name on blur', async () => {
            const onUpdate = vi.fn();

            renderApp({ label, onUpdate });

            const input = screen.getByLabelText(/Label name/i);
            await userEvent.clear(input);
            await userEvent.type(input, 'new name');
            await userEvent.tab();

            expect(onUpdate).toHaveBeenCalledWith(label.id, expect.objectContaining({ name: 'new name' }));
        });

        it('calls onUpdate on Enter key', async () => {
            const onUpdate = vi.fn();

            renderApp({ label, onUpdate });

            const input = screen.getByLabelText(/Label name/i);
            await userEvent.clear(input);
            await userEvent.type(input, 'new name{Enter}');

            expect(onUpdate).toHaveBeenCalledWith(label.id, expect.objectContaining({ name: 'new name' }));
        });

        it('does not call onUpdate when name has validation error', async () => {
            const onUpdate = vi.fn();
            const validateName = vi.fn(() => 'Name already exists');

            renderApp({ label, onUpdate, validateName });

            const input = screen.getByLabelText(/Label name/i);
            await userEvent.clear(input);
            await userEvent.type(input, 'duplicate');
            await userEvent.tab();

            expect(onUpdate).not.toHaveBeenCalled();
        });

        it('does not call onUpdate when name is empty', async () => {
            const onUpdate = vi.fn();

            renderApp({ label, onUpdate });

            const input = screen.getByLabelText(/Label name/i);
            await userEvent.clear(input);
            await userEvent.tab();

            expect(onUpdate).not.toHaveBeenCalled();
        });

        it('does not call onUpdate when name has not changed', async () => {
            const onUpdate = vi.fn();

            renderApp({ label, onUpdate });

            const input = screen.getByLabelText(/Label name/i);
            await userEvent.clear(input);
            await userEvent.type(input, label.name);
            await userEvent.tab();

            expect(onUpdate).not.toHaveBeenCalled();
        });
    });
});
