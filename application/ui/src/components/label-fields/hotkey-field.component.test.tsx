// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { HotkeyField } from './hotkey-field.component';

describe('HotkeyField', () => {
    const mockOnHotkeyChange = vi.fn();

    beforeEach(() => {
        mockOnHotkeyChange.mockClear();
    });

    it('renders with placeholder when hotkey is null', () => {
        render(<HotkeyField hotkey={null} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        expect(input).toHaveAttribute('placeholder', 'Hotkey');
        expect(input).toHaveValue('');
    });

    it('displays formatted hotkey value', () => {
        render(<HotkeyField hotkey='ctrl+s' onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        expect(input).toHaveValue('CTRL+S');
    });

    it('captures simple key press', () => {
        render(<HotkeyField hotkey={null} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        fireEvent.keyDown(input, { key: 'a' });

        expect(mockOnHotkeyChange).toHaveBeenCalledWith('a');
    });

    it('captures key with ctrl modifier', () => {
        render(<HotkeyField hotkey={null} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        fireEvent.keyDown(input, { key: 's', ctrlKey: true });

        expect(mockOnHotkeyChange).toHaveBeenCalledWith('ctrl+s');
    });

    it('captures key with multiple modifiers', () => {
        render(<HotkeyField hotkey={null} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        fireEvent.keyDown(input, { key: 's', ctrlKey: true, shiftKey: true });

        expect(mockOnHotkeyChange).toHaveBeenCalledWith('ctrl+shift+s');
    });

    it('ignores standalone Control key press', () => {
        render(<HotkeyField hotkey={null} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        fireEvent.keyDown(input, { key: 'Control' });

        expect(mockOnHotkeyChange).not.toHaveBeenCalled();
    });

    it('ignores standalone Alt key press', () => {
        render(<HotkeyField hotkey={null} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        fireEvent.keyDown(input, { key: 'Alt' });

        expect(mockOnHotkeyChange).not.toHaveBeenCalled();
    });

    it('ignores standalone Shift key press', () => {
        render(<HotkeyField hotkey={null} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        fireEvent.keyDown(input, { key: 'Shift' });

        expect(mockOnHotkeyChange).not.toHaveBeenCalled();
    });

    it('ignores standalone Meta key press', () => {
        render(<HotkeyField hotkey={null} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        fireEvent.keyDown(input, { key: 'Meta' });

        expect(mockOnHotkeyChange).not.toHaveBeenCalled();
    });

    it('shows error message when provided', () => {
        render(<HotkeyField hotkey='a' onHotkeyChange={mockOnHotkeyChange} errorMessage='That hotkey is already in use' />);

        expect(screen.getByText('That hotkey is already in use')).toBeInTheDocument();
    });

    it('does not show error when errorMessage is undefined', () => {
        render(<HotkeyField hotkey='c' onHotkeyChange={mockOnHotkeyChange} />);

        expect(screen.queryByText('That hotkey is already in use')).not.toBeInTheDocument();
    });

    it('handles undefined hotkey same as null', () => {
        render(<HotkeyField hotkey={undefined} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        expect(input).toHaveValue('');
    });

    it('captures alt modifier', () => {
        render(<HotkeyField hotkey={null} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        fireEvent.keyDown(input, { key: 'a', altKey: true });

        expect(mockOnHotkeyChange).toHaveBeenCalledWith('alt+a');
    });

    it('captures meta modifier', () => {
        render(<HotkeyField hotkey={null} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        fireEvent.keyDown(input, { key: 'a', metaKey: true });

        expect(mockOnHotkeyChange).toHaveBeenCalledWith('meta+a');
    });

    it('builds modifier order as ctrl+meta+alt+shift', () => {
        render(<HotkeyField hotkey={null} onHotkeyChange={mockOnHotkeyChange} />);

        const input = screen.getByRole('textbox', { name: 'Hotkey input' });
        fireEvent.keyDown(input, { key: 'a', ctrlKey: true, metaKey: true, altKey: true, shiftKey: true });

        expect(mockOnHotkeyChange).toHaveBeenCalledWith('ctrl+meta+alt+shift+a');
    });
});
