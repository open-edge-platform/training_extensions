// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { TestProviders } from '../../providers';
import { RequiredTextField } from './required-text-field.component';

describe('RequiredTextField', () => {
    const errorMessage = 'This field is required';

    it('does not display the error message before the input is interacted with', async () => {
        render(
            <TestProviders>
                <RequiredTextField label='Label' errorMessage={errorMessage} aria-label='Required Text Field' />
            </TestProviders>
        );

        expect(screen.queryByAltText(errorMessage)).not.toBeInTheDocument();
    });

    it('shows error message when field has been touched and is empty', async () => {
        render(
            <TestProviders>
                <RequiredTextField label='Label' errorMessage={errorMessage} aria-label='Required Text Field' />
            </TestProviders>
        );

        const input = screen.getByLabelText(/Required Text Field/i);

        await userEvent.type(input, 'test');
        await userEvent.clear(input);

        expect(screen.getByText(errorMessage)).toBeVisible();
    });
});
