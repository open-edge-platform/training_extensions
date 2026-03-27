// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { DataCollection } from './data-collection.component';

describe('DataCollection capture rate fields', () => {
    const renderApp = () => {
        const pipelinePatchSpy = vi.fn();

        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                return HttpResponse.json(
                    getMockedPipeline({
                        data_collection: {
                            max_dataset_size: 500,
                            policies: [
                                {
                                    type: 'fixed_rate',
                                    rate: 12,
                                    enabled: true,
                                },
                                {
                                    type: 'confidence_threshold',
                                    confidence_threshold: 0.5,
                                    min_sampling_interval: 2.5,
                                    enabled: false,
                                },
                            ],
                        },
                    })
                );
            }),
            http.patch('/api/projects/{project_id}/pipeline', async ({ request }) => {
                pipelinePatchSpy(await request.json());

                return HttpResponse.json({});
            })
        );

        render(<DataCollection />);

        return pipelinePatchSpy;
    };

    beforeEach(() => {
        vi.resetAllMocks();
    });

    it('renders Frames and Seconds fields for capture rate', async () => {
        renderApp();

        expect(await screen.findByLabelText('Frames')).toBeVisible();
        expect(screen.getByLabelText('Seconds')).toBeVisible();
        expect(screen.queryByLabelText('Rate')).not.toBeInTheDocument();
    });

    it('patches fixed_rate using frames divided by seconds', async () => {
        const pipelinePatchSpy = renderApp();

        const framesInput = await screen.findByLabelText('Frames');
        const secondsInput = screen.getByLabelText('Seconds');

        await userEvent.clear(framesInput);
        await userEvent.type(framesInput, '6');
        framesInput.blur();

        await userEvent.clear(secondsInput);
        await userEvent.type(secondsInput, '3');
        secondsInput.blur();

        await waitFor(() => {
            expect(pipelinePatchSpy).toHaveBeenCalled();
        });

        const lastMutationPayload = pipelinePatchSpy.mock.calls.at(-1)?.[0];
        const fixedRatePolicy = lastMutationPayload.data_collection.policies.find(
            (policy: { type: string }) => policy.type === 'fixed_rate'
        );

        expect(fixedRatePolicy.rate).toBe(2);
    });
});
