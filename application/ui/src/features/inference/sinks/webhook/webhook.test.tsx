// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import userEvent from '@testing-library/user-event';
import { render, screen, waitFor } from 'test-utils/render';

import { useSinkAction } from '../hooks/use-sink-action.hook';
import { WebhookHttpMethod, WebhookSinkConfig } from '../utils';
import { Webhook } from './webhook.component';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

vi.mock('../hooks/use-sink-action.hook');

const mockedConfig: WebhookSinkConfig = {
    id: '1',
    name: 'webhook test',
    timeout: 0,
    sink_type: 'webhook',
    rate_limit: 0,
    webhook_url: 'localHost',
    http_method: WebhookHttpMethod.POST,
    output_formats: [],
    headers: {},
};

describe('Webhook', () => {
    it('disables the Apply button when loading', () => {
        vi.mocked(useSinkAction).mockReturnValue([mockedConfig, vi.fn(), true]);

        render(<Webhook />);

        expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
    });

    it('calls submit action when Apply button is clicked', async () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSinkAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(<Webhook config={mockedConfig} />);

        userEvent.click(screen.getByRole('button', { name: 'Apply' }));

        await waitFor(() => expect(mockedSubmitAction).toHaveBeenCalled());
    });

    it('renders fields with correct values from config', () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSinkAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(<Webhook config={mockedConfig} />);

        expect(useSinkAction).toHaveBeenCalledWith({
            config: mockedConfig,
            isNewSink: false,
            bodyFormatter: expect.anything(),
        });

        expect(screen.getByRole('textbox', { name: /Id/i, hidden: true })).toHaveValue(mockedConfig.id);
        expect(screen.getByRole('textbox', { name: /Name/i })).toHaveValue(mockedConfig.name);
        expect(screen.getByRole('textbox', { name: /Rate Limit/i })).toHaveValue(String(mockedConfig.rate_limit));
        expect(screen.getByRole('textbox', { name: /Webhook URL/i })).toHaveValue(mockedConfig.webhook_url);
        expect(screen.getByRole('textbox', { name: /Timeout/i })).toHaveValue(String(mockedConfig.timeout));
        expect(screen.getByRole('button', { name: /HTTP Method/i })).toHaveTextContent(WebhookHttpMethod.POST);

        screen.getAllByRole<HTMLInputElement>('checkbox').forEach((checkbox) => {
            // @ts-expect-error: checkbox.value is string, but output_formats expects specific string literals
            if (mockedConfig.output_formats.includes(checkbox.value)) {
                expect(checkbox).toBeChecked();
            } else {
                expect(checkbox).not.toBeChecked();
            }
        });

        expect(screen.getByRole('button', { name: 'Apply' })).toBeEnabled();
    });
});
