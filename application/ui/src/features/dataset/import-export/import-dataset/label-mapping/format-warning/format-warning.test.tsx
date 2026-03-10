// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../../../../api/utils';
import { server } from '../../../../../../msw-node-setup';
import { FormatWarning } from './format-warning.component';

describe('FormatWarning', () => {
    const mockedProject = getMockedProject({ id: 'project-123' });

    it('renders bounding box to polygons warning for instance segmentation project', async () => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json({
                    ...mockedProject,
                    task: { ...mockedProject.task, task_type: 'instance_segmentation' },
                });
            })
        );

        render(<FormatWarning annotationType={'bounding_box'} />);

        expect(await screen.findByText(/Imported dataset uses bounding box annotations/i)).toBeInTheDocument();
    });

    it('renders polygon to bounding boxes warning for detection project', async () => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json({
                    ...mockedProject,
                    task: { ...mockedProject.task, task_type: 'detection' },
                });
            })
        );

        render(<FormatWarning annotationType={'polygon'} />);

        expect(await screen.findByText(/Imported dataset uses polygon annotations/i)).toBeInTheDocument();
    });
});
