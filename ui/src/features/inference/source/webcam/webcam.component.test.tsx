// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';

import { TestProviders } from '../../../../providers';
import { errorMessage } from './permissions-error.component';
import { useVideoDevices } from './use-video-devices.hook';
import { UserCameraPermission } from './util';
import { Webcam } from './webcam.component';

vi.mock('./use-video-devices.hook', () => ({
    useVideoDevices: vi.fn(),
}));

const renderApp = ({
    videoDevices = [],
    setSelectedDeviceId = vi.fn(),
    userPermissions = UserCameraPermission.PENDING,
}: {
    videoDevices?: MediaDeviceInfo[];
    setSelectedDeviceId?: () => void;
    userPermissions?: UserCameraPermission;
}) => {
    vi.mocked(useVideoDevices).mockReturnValue({
        videoDevices,
        selectedDeviceId: undefined,
        setSelectedDeviceId,
        userPermissions,
    });

    render(
        <TestProviders>
            <Webcam />
        </TestProviders>
    );
};

describe('Webcam', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders loading when permissions are pending', () => {
        renderApp({ userPermissions: UserCameraPermission.PENDING });
        expect(screen.getByLabelText('permissions pending')).toBeInTheDocument();
    });

    it('renders PermissionError when permissions are denied', () => {
        renderApp({ userPermissions: UserCameraPermission.DENIED });
        expect(screen.getByText(errorMessage)).toBeVisible();
    });

    it('renders Picker with devices when permissions are granted', async () => {
        const mockSetSelectedDeviceId = vi.fn();

        const videoDevices: MediaDeviceInfo[] = [
            { deviceId: '1', label: 'Camera 1', groupId: 'group1', kind: 'videoinput', toJSON: () => {} },
            { deviceId: '2', label: 'Camera 2', groupId: 'group2', kind: 'videoinput', toJSON: () => {} },
        ];

        renderApp({
            videoDevices,
            setSelectedDeviceId: mockSetSelectedDeviceId,
            userPermissions: UserCameraPermission.GRANTED,
        });

        userEvent.click(screen.getByLabelText('devices'));

        expect(await screen.findByRole('option', { name: videoDevices[0].label })).toBeVisible();
        userEvent.click(screen.getByRole('option', { name: videoDevices[1].label }));

        await waitFor(() => {
            expect(mockSetSelectedDeviceId).toHaveBeenCalledWith(videoDevices[1].deviceId);
        });
    });
});
