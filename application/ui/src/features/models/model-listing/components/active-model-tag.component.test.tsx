// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';

import { ActiveModelTag } from './active-model-tag.component';

describe('ActiveModelTag', () => {
    it('renders active tag with icon and text', () => {
        const { container } = render(<ActiveModelTag />);

        expect(screen.getByText('Active')).toBeInTheDocument();
        expect(container.querySelector('svg')).toBeInTheDocument();
        expect(container.firstChild).toBeInTheDocument();
    });
});
