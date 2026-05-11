// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedLabel } from 'mocks/mock-labels';
import { getMockedVideoFrame } from 'mocks/mock-media';
import { render } from 'test-utils/render';

import { VideoToolbar } from './video-toolbar.component';

const mockVideoFrame = getMockedVideoFrame({
    fps: 60,
    frame_count: 600,
    frame_number: 42,
    frame_stride: 1,
    duration: 10,
});

vi.mock('../video-player-provider.component', () => ({
    useVideoPlayer: () => ({
        videoFrame: mockVideoFrame,
        step: 1,
        changeStep: vi.fn(),
        isMuted: false,
        toggleMute: vi.fn(),
        playbackRate: 1,
        changePlaybackRate: vi.fn(),
        videoControls: {
            isPlaying: false,
            play: vi.fn(),
            pause: vi.fn(),
            goto: vi.fn(),
            previousFrame: vi.fn(),
            nextFrame: vi.fn(),
            canSelectPreviousFrame: true,
            canSelectNextFrame: true,
        },
    }),
}));

vi.mock('../../../../hooks/use-project-labels.hook', () => ({
    useProjectLabels: () => [getMockedLabel({ id: 'label-1', name: 'Cat' })],
}));

vi.mock('../../predictions-setup-provider.component', async (importOriginal) => ({
    ...(await importOriginal()),
    usePredictionSetup: () => ({
        selectedModelId: 'model-1',
        selectedModel: { id: 'variant-1', name: 'Test Model [FP32]', modelId: 'model-1' },
        changeSelectedModelId: vi.fn(),
        selectableModels: [{ id: 'variant-1', name: 'Test Model [FP32]', modelId: 'model-1' }],
    }),
}));

const toggleToolbar = async () => {
    await userEvent.click(screen.getByRole('button', { name: /Expand|Collapse toolbar/ }));
};

describe('VideoToolbar', () => {
    describe('collapsed state (default)', () => {
        it('renders playback controls', () => {
            render(<VideoToolbar mode='annotation' />);

            expect(screen.getByRole('button', { name: 'Play video' })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Go to previous frame' })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Go to next frame' })).toBeInTheDocument();
        });

        it('renders the mute button', () => {
            render(<VideoToolbar mode='annotation' />);

            expect(screen.getByRole('button', { name: 'Mute audio' })).toBeInTheDocument();
        });

        it('renders the video duration', () => {
            render(<VideoToolbar mode='annotation' />);

            expect(screen.getByRole('generic', { name: 'Video duration' })).toBeInTheDocument();
        });

        it('renders the video player slider', () => {
            render(<VideoToolbar mode='annotation' />);

            expect(screen.getByRole('slider', { name: 'Video timeline' })).toBeInTheDocument();
        });

        it('renders the expand toolbar button', () => {
            render(<VideoToolbar mode='annotation' />);

            expect(screen.getByRole('button', { name: 'Expand toolbar' })).toBeInTheDocument();
        });

        it('does not render the "Frames" label', () => {
            render(<VideoToolbar mode='annotation' />);

            expect(screen.queryByText('Frames')).not.toBeInTheDocument();
        });

        it('does not render the frame step toggle', () => {
            render(<VideoToolbar mode='annotation' />);

            expect(screen.queryByRole('button', { name: 'Toggle frame mode' })).not.toBeInTheDocument();
        });

        it('does not render the playback speed control', () => {
            render(<VideoToolbar mode='annotation' />);

            expect(screen.queryByRole('button', { name: 'Change playback speed' })).not.toBeInTheDocument();
        });

        it('does not render the current frame info text', () => {
            render(<VideoToolbar mode='annotation' />);

            expect(screen.queryByText(/Current frame/)).not.toBeInTheDocument();
        });

        it('does not render the video annotator timeline', () => {
            render(<VideoToolbar mode='annotation' />);

            expect(screen.queryByRole('grid', { name: 'Video timeline' })).not.toBeInTheDocument();
        });
    });

    describe('expanded state', () => {
        it('renders the "Frames" label', async () => {
            render(<VideoToolbar mode='annotation' />);
            await toggleToolbar();

            expect(screen.getByText('Frames')).toBeInTheDocument();
        });

        it('renders playback controls', async () => {
            render(<VideoToolbar mode='annotation' />);
            await toggleToolbar();

            expect(screen.getByRole('button', { name: 'Play video' })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Go to previous frame' })).toBeInTheDocument();
            expect(screen.getByRole('button', { name: 'Go to next frame' })).toBeInTheDocument();
        });

        it('renders the mute button', async () => {
            render(<VideoToolbar mode='annotation' />);
            await toggleToolbar();

            expect(screen.getByRole('button', { name: 'Mute audio' })).toBeInTheDocument();
        });

        it('renders the video duration', async () => {
            render(<VideoToolbar mode='annotation' />);
            await toggleToolbar();

            expect(screen.getByRole('generic', { name: 'Video duration' })).toBeInTheDocument();
        });

        it('renders the frame step toggle', async () => {
            render(<VideoToolbar mode='annotation' />);
            await toggleToolbar();

            expect(screen.getByRole('button', { name: 'Toggle frame mode' })).toBeInTheDocument();
        });

        it('renders the playback speed control', async () => {
            render(<VideoToolbar mode='annotation' />);
            await toggleToolbar();

            expect(screen.getByRole('button', { name: 'Change playback speed' })).toBeInTheDocument();
        });

        it('renders the current frame and total frames info', async () => {
            render(<VideoToolbar mode='annotation' />);
            await toggleToolbar();

            expect(
                screen.getByText(
                    `Current frame: ${mockVideoFrame.frame_number} / Total frames: ${mockVideoFrame.frame_count - 1}`
                )
            ).toBeInTheDocument();
        });

        it('renders the video annotator timeline grid', async () => {
            render(<VideoToolbar mode='annotation' />);
            await toggleToolbar();

            expect(await screen.findByRole('grid', { name: 'Video timeline' })).toBeInTheDocument();
        });

        it('renders the collapse toolbar button', async () => {
            render(<VideoToolbar mode='annotation' />);
            await toggleToolbar();

            expect(screen.getByRole('button', { name: 'Collapse toolbar' })).toBeInTheDocument();
        });
    });

    describe('expand / collapse toggle', () => {
        it('expands when the expand button is clicked', async () => {
            render(<VideoToolbar mode='annotation' />);

            await toggleToolbar();

            await waitFor(() => {
                expect(screen.getByText('Frames')).toBeInTheDocument();
            });
        });

        it('collapses back when the collapse button is clicked', async () => {
            render(<VideoToolbar mode='annotation' />);

            await toggleToolbar();
            await toggleToolbar();

            expect(screen.queryByText('Frames')).not.toBeInTheDocument();
        });
    });
});
