// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@test-utils/render';
import userEvent from '@testing-library/user-event';

import { RequiredTextField } from './required-text-field.component';

describe('RequiredTextField', () => {
    const errorMessage = 'This field is required';

    it('does not display the error message before the input is interacted with', async () => {
        render(<RequiredTextField label='Label' errorMessage={errorMessage} aria-label='Required Text Field' />);

        expect(screen.queryByAltText(errorMessage)).not.toBeInTheDocument();
    });

    it('shows error message when field has been touched and is empty', async () => {
        render(<RequiredTextField label='Label' errorMessage={errorMessage} aria-label='Required Text Field' />);

        const input = screen.getByLabelText(/Required Text Field/i);

        await userEvent.type(input, 'test');
        await userEvent.clear(input);

        expect(screen.getByText(errorMessage)).toBeVisible();
    });
});
