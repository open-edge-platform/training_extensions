// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from 'test-utils/render';

import { MediaThumbnail } from './media-thumbnail.component';

describe('MediaThumbnail', () => {
    it('calls onClick when image is clicked', async () => {
        const mockedClick = vi.fn();
        render(<MediaThumbnail url='test-image.jpg' alt='Test Image' onClick={mockedClick} item={{ type: 'image' }} />);

        await userEvent.click(screen.getByRole('img', { name: 'Test Image' }));
        await waitFor(() => expect(mockedClick).toHaveBeenCalled());
    });

    it('calls onDoubleClick when image is double-clicked', async () => {
        const mockedDblClick = vi.fn();
        render(
            <MediaThumbnail
                url='test-image.jpg'
                alt='Test Image'
                onDoubleClick={mockedDblClick}
                item={{ type: 'image' }}
            />
        );

        await userEvent.dblClick(screen.getByRole('img', { name: 'Test Image' }));
        await waitFor(() => expect(mockedDblClick).toHaveBeenCalled());
    });

    it('displays frames count when item is a video', async () => {
        const mockedClick = vi.fn();
        render(
            <MediaThumbnail
                url='test-video.mp4'
                alt='Test Image'
                onClick={mockedClick}
                item={{ type: 'video', frame_count: 3600, annotated_frame_count: 10, duration: 60 }}
            />
        );

        expect(screen.getByText('00:01:00')).toBeInTheDocument();
    });
});
