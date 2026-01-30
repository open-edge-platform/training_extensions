// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';

import { NewProjectLink } from './new-project-link.component';

// Helper to render with router context
const renderWithRouter = (component: React.ReactElement) => {
    return render(<BrowserRouter>{component}</BrowserRouter>);
};

describe('NewProjectLink', () => {
    it('renders a clickable link with icon and text to new project path', () => {
        const { container } = renderWithRouter(<NewProjectLink />);

        const link = screen.getByRole('link', { name: /create project/i });
        expect(link).toBeInTheDocument();
        expect(link).toHaveAttribute('href', '/projects/new');

        const svg = container.querySelector('svg');
        const text = screen.getByText('Create project');
        expect(svg).toBeInTheDocument();
        expect(text).toBeInTheDocument();
    });
});
