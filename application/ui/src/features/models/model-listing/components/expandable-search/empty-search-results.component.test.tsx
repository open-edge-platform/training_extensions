// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';

import { EmptySearchResults } from './empty-search-results.component';

describe('EmptySearchResults', () => {
    it('renders with correct content and structure', () => {
        const { container } = render(<EmptySearchResults />);

        const heading = screen.getByRole('heading', { name: 'No models found' });
        expect(heading).toBeInTheDocument();
        expect(heading.tagName).toBe('H3');
        expect(container.querySelector('svg')).toBeInTheDocument();
        expect(container.firstChild).toBeInTheDocument();
    });

    it('renders as a static component with no interactions', () => {
        const { container } = render(<EmptySearchResults />);

        const buttons = container.querySelectorAll('button');
        const links = container.querySelectorAll('a');

        expect(buttons).toHaveLength(0);
        expect(links).toHaveLength(0);
    });
});
