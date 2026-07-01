// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { ViewModes } from '@geti-ui/ui';
import { screen, waitFor, waitForElementToBeRemoved } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedMediaImage } from 'mocks/mock-media';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { render } from 'test-utils/render';

import { http } from '../../../api/utils';
import { server } from '../../../msw-node-setup';
import { SelectedDataProvider, useSelectedData } from '../providers/selected-data-provider.component';
import { Gallery } from './gallery.component';

const uploadMediaMock = vi.fn();

let dropFiles: (files: File[]) => void = () => {};

vi.mock('../../../shared/drop-zone.utils', () => ({
    getFilesFromDropEvent: ({ files }: { files: File[] }) => Promise.resolve(files),
}));

vi.mock('../../../components/virtualizer-grid-layout/virtualizer-grid-layout.component', () => ({
    VirtualizerGridLayout: ({
        items,
        contentItem,
    }: {
        items: Array<{ id: string }>;
        contentItem: (item: unknown) => ReactNode;
    }) => (
        <>
            {items.map((item) => (
                <div key={item.id}>{contentItem(item)}</div>
            ))}
        </>
    ),
}));

vi.mock('@geti-ui/ui', async (importOriginal) => {
    const actual = await importOriginal<typeof import('@geti-ui/ui')>();
    return {
        ...actual,
        AriaDropZone: ({
            children,
            onDrop,
        }: {
            children: ReactNode | ((state: { isDropTarget: boolean }) => ReactNode);
            onDrop?: (event: { files: File[] }) => void;
        }) => {
            dropFiles = (files) => onDrop?.({ files });
            return <>{typeof children === 'function' ? children({ isDropTarget: false }) : children}</>;
        },
    };
});

vi.mock('../api/use-media-upload', () => ({
    useMediaUpload: () => ({
        uploadMedia: uploadMediaMock,
        uploadProgress: { total: 0, completed: 0, succeeded: 0, failed: 0, isUploading: false },
    }),
}));

vi.mock('./hooks/use-select-dataset-item.hook', () => ({
    useSelectDatasetItem: () => ({
        selectedMediaItem: null,
        onSelectedMediaItemChange: vi.fn(),
    }),
}));

vi.mock('hooks/use-project-identifier.hook', () => ({
    useProjectIdentifier: () => 'project-123',
}));

describe('Gallery drag-and-drop upload', () => {
    const renderGallery = async () => {
        server.use(
            http.get('/api/projects/{project_id}', () => {
                return HttpResponse.json(getMockedProject({ id: 'project-123' }));
            })
        );

        render(
            <SelectedDataProvider>
                <Gallery
                    items={[]}
                    viewMode={ViewModes.LARGE}
                    isPending={false}
                    hasActiveFilter={false}
                    isFetchingNextPage={false}
                    fetchNextPage={vi.fn()}
                    isMediaItemReviewedById={() => false}
                />
            </SelectedDataProvider>
        );

        await waitForElementToBeRemoved(() => screen.queryByRole('progressbar'));
    };

    beforeEach(() => {
        uploadMediaMock.mockReset();
    });

    it('uploads supported files without showing an error toast', async () => {
        await renderGallery();

        dropFiles([
            new File([''], 'photo.png', { type: 'image/png' }),
            new File([''], 'clip.mp4', { type: 'video/mp4' }),
        ]);

        await waitFor(() => expect(uploadMediaMock).toHaveBeenCalledTimes(1));
        expect(uploadMediaMock.mock.calls[0][0].map((f: File) => f.name)).toEqual(['photo.png', 'clip.mp4']);
        expect(screen.queryByLabelText('toast')).not.toBeInTheDocument();
    });
});

describe('Gallery item deletion and selection', () => {
    const item = getMockedMediaImage({ id: 'item-1' });

    const GalleryWithSelectionCount = ({ items }: { items: (typeof item)[] }) => {
        const { selectedKeys } = useSelectedData();
        const count = selectedKeys instanceof Set ? selectedKeys.size : 0;

        return (
            <>
                {count > 0 && <p>{count} selected</p>}
                <Gallery
                    items={items}
                    viewMode={ViewModes.LARGE}
                    isPending={false}
                    hasActiveFilter={false}
                    isFetchingNextPage={false}
                    fetchNextPage={vi.fn()}
                    isMediaItemReviewedById={() => false}
                />
            </>
        );
    };

    const renderGalleryWithItems = async (items: (typeof item)[]) => {
        server.use(
            http.get('/api/projects/{project_id}', () => HttpResponse.json(getMockedProject({ id: 'project-123' }))),
            http.delete('/api/projects/{project_id}/dataset/media', () => new HttpResponse(null, { status: 204 }))
        );

        render(
            <SelectedDataProvider>
                <GalleryWithSelectionCount items={items} />
            </SelectedDataProvider>
        );

        await waitForElementToBeRemoved(() => screen.queryByRole('progressbar'));
    };

    it('does not show a selected count after deleting a non-selected item', async () => {
        const user = userEvent.setup();
        await renderGalleryWithItems([item]);

        expect(screen.queryByText(/\d+ selected/)).not.toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Media actions' }));
        await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));
        await user.click(await screen.findByRole('button', { name: 'Confirm' }));
        await screen.findByText('1 item deleted successfully');

        expect(screen.queryByText(/\d+ selected/)).not.toBeInTheDocument();
    });

    it('removes the deleted item from the selected count', async () => {
        const user = userEvent.setup();
        await renderGalleryWithItems([item]);

        await user.click(screen.getByRole('checkbox', { name: `Select media item ${item.id}` }));
        expect(screen.getByText('1 selected')).toBeInTheDocument();

        await user.click(screen.getByRole('button', { name: 'Media actions' }));
        await user.click(await screen.findByRole('menuitem', { name: 'Delete' }));
        await user.click(await screen.findByRole('button', { name: 'Confirm' }));
        await screen.findByText('1 item deleted successfully');

        expect(screen.queryByText(/\d+ selected/)).not.toBeInTheDocument();
    });
});
