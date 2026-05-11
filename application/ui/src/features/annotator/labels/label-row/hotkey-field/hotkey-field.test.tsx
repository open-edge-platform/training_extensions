// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { HotkeyField } from './hotkey-field.component';

describe('Hotkey edition field', () => {
    const onChangeMock = vi.fn();

    const renderApp = (value = 'control+a'): HTMLElement => {
        render(
            <HotkeyField hasAutoFocus={true} value={value} onChange={onChangeMock} aria-label={'test hotkey field'} />
        );

        const hotkeyField = screen.getByRole('textbox', { name: 'test hotkey field' });
        fireEvent.click(hotkeyField);
        expect(hotkeyField).toHaveValue(value);

        return hotkeyField;
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('check if edited hotkey has visible new value', async () => {
        const user = userEvent.setup();
        renderApp();

        await user.keyboard('{s>}d{/s}');

        await waitFor(() => {
            expect(onChangeMock).toHaveBeenCalledWith('S+D');
        });
    });

    it('check if hotkey is limited to 2 keys', async () => {
        const user = userEvent.setup();
        renderApp();

        await user.keyboard('{q>}{e>}r{/q}{/e}');
        expect(onChangeMock).toHaveBeenCalledWith('E+R');
    });

    it('does not add multiple modifiers', async () => {
        const user = userEvent.setup();
        renderApp();

        await user.keyboard('{shift}{ctrl/}');
        expect(onChangeMock).toHaveBeenCalledWith('SHIFT');

        await user.keyboard('{Alt}{Meta/}');
        expect(onChangeMock).toHaveBeenCalledWith('ALT');
    });

    it('add "Shift+number" as numerical value', async () => {
        const user = userEvent.setup();
        renderApp();

        await user.keyboard('{Shift>}1{/Shift}');
        expect(onChangeMock).toHaveBeenCalledWith('SHIFT+1');

        await user.keyboard('{Shift>}9{/Shift}');
        expect(onChangeMock).toHaveBeenCalledWith('SHIFT+9');
    });

    it('calls onEnter when Enter key is pressed while focused', async () => {
        const onEnterMock = vi.fn();
        const user = userEvent.setup();

        render(
            <HotkeyField
                hasAutoFocus={true}
                value={'control+a'}
                onChange={onChangeMock}
                onEnter={onEnterMock}
                aria-label={'test hotkey field'}
            />
        );

        const hotkeyField = screen.getByRole('textbox', { name: 'test hotkey field' });
        fireEvent.click(hotkeyField);

        await user.keyboard('{Enter}');
        expect(onEnterMock).toHaveBeenCalledTimes(1);
    });

    it('does not throw when onEnter is not provided and Enter is pressed', async () => {
        const user = userEvent.setup();
        const hotkeyField = renderApp();

        await expect(user.keyboard('{Enter}')).resolves.not.toThrow();
        expect(hotkeyField).toHaveValue('control+a');
    });
});
