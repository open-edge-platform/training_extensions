// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, screen } from '@testing-library/react';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { acceptedExtensions, AddMediaButton } from './add-media-button.component';

describe('AddMediaButton', () => {
    it('calls onFilesSelected correctly', () => {
        const mockOnFilesSelected = vi.fn();
        const mockFile = new File(['file content'], 'test-image.jpg', {
            type: 'image/jpeg',
            lastModified: Date.now(),
        });

        render(
            <AddMediaButton
                onFilesSelected={mockOnFilesSelected}
                project={getMockedProject({
                    task: {
                        task_type: 'detection',
                        labels: [getMockedLabel()],
                        exclusive_labels: true,
                    },
                })}
            />
        );

        const input = screen.getByLabelText(/Upload media files/);
        fireEvent.change(input, { target: { files: [mockFile] } });

        expect(mockOnFilesSelected).toHaveBeenCalledWith([mockFile]);
    });

    it('sets the expected accepted file extensions', () => {
        render(<AddMediaButton onFilesSelected={vi.fn()} project={getMockedProject()} />);

        const input = screen.getByLabelText(/Upload media files/);

        expect(input).toHaveAttribute('accept', acceptedExtensions);
    });

    it('opens file picker when button is clicked', () => {
        const mockOnFilesSelected = vi.fn();

        render(<AddMediaButton onFilesSelected={mockOnFilesSelected} project={getMockedProject()} />);

        const button = screen.getByRole('button', { name: /Upload media/ });
        const input = screen.getByLabelText(/Upload media files/) as HTMLInputElement;

        const clickSpy = vi.spyOn(input, 'click');
        fireEvent.click(button);

        expect(clickSpy).toHaveBeenCalled();
    });

    it('disables button when isDisabled prop is true', () => {
        render(<AddMediaButton onFilesSelected={vi.fn()} isDisabled project={getMockedProject()} />);

        const button = screen.getByRole('button', { name: /Upload media/ });

        expect(button).toBeDisabled();
    });

    it('opens bulk label assignment dialog for classification project', async () => {
        const mockOnFilesSelected = vi.fn();
        const mockFile = new File(['file content'], 'test-image.jpg', {
            type: 'image/jpeg',
            lastModified: Date.now(),
        });

        const project = getMockedProject({
            task: {
                task_type: 'classification',
                exclusive_labels: true,
                labels: [getMockedLabel()],
            },
        });

        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(project);
            })
        );

        render(<AddMediaButton onFilesSelected={mockOnFilesSelected} project={project} />);

        const input = screen.getByLabelText(/Upload media files/);
        fireEvent.change(input, { target: { files: [mockFile] } });

        expect(await screen.findByRole('heading', { name: 'Assign the label to the images' })).toBeInTheDocument();

        expect(mockOnFilesSelected).not.toHaveBeenCalledWith([mockFile]);
    });
});
