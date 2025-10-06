// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import userEvent from '@testing-library/user-event';
import { render, screen, waitFor } from 'test-utils/render';

import { useSinkAction } from '../hooks/use-sink-action.hook';
import { MqttSinkConfig } from '../utils';
import { Mqtt } from './mqtt.component';

vi.mock('react-router', async (importOriginal) => {
    const actual = await importOriginal<typeof import('react-router')>();
    return {
        ...actual,
        useParams: vi.fn(() => ({ projectId: '123' })),
    };
});

vi.mock('../hooks/use-sink-action.hook');

const mockedConfig: MqttSinkConfig = {
    id: '1',
    name: 'Test Folder',
    output_formats: [],
    sink_type: 'mqtt',
    broker_host: '',
    broker_port: 0,
    topic: '',
    auth_required: true,
    rate_limit: 0,
};

describe('Mqtt', () => {
    it('disables the Apply button when loading', () => {
        vi.mocked(useSinkAction).mockReturnValue([mockedConfig, vi.fn(), true]);

        render(<Mqtt />);

        expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
    });

    it('calls submit action when Apply button is clicked', async () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSinkAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(<Mqtt config={mockedConfig} />);

        userEvent.click(screen.getByRole('button', { name: 'Apply' }));

        await waitFor(() => expect(mockedSubmitAction).toHaveBeenCalled());
    });

    it('renders fields with correct values from config', () => {
        const mockedSubmitAction = vi.fn();
        vi.mocked(useSinkAction).mockReturnValue([mockedConfig, mockedSubmitAction, false]);

        render(<Mqtt config={mockedConfig} />);

        expect(useSinkAction).toHaveBeenCalledWith({
            config: mockedConfig,
            isNewSink: false,
            bodyFormatter: expect.anything(),
        });

        expect(screen.getByRole('textbox', { name: /Id/i, hidden: true })).toHaveValue(mockedConfig.id);
        expect(screen.getByRole('textbox', { name: /Name/i })).toHaveValue(mockedConfig.name);
        expect(screen.getByRole('textbox', { name: /Broker Host/i })).toHaveValue(mockedConfig.broker_host);
        expect(screen.getByRole('textbox', { name: /Topic/i })).toHaveValue(mockedConfig.topic);
        expect(screen.getByRole('textbox', { name: /Broker Port/i })).toHaveValue(String(mockedConfig.broker_port));
        expect(screen.getByRole('textbox', { name: /Rate Limit/i })).toHaveValue(String(mockedConfig.rate_limit));
        expect(screen.getByLabelText('Require Authentication')).toBeChecked();

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
