// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import '@wessberg/pointer-events';

import { ReactNode, useEffect } from 'react';

import { fireEvent, screen, waitFor } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';

import { Rect, Shape } from '../../../../core/annotations/shapes.interface';
import { ShapeType } from '../../../../core/annotations/shapetype.enum';
import { getMockedImage } from '../../../../test-utils/utils';
import { LabelsShortcuts } from '../../components/main-content/labels-shortcuts/labels-shortcuts.component';
import { ToolType } from '../../core/annotation-tool-context.interface';
import {
    AnnotationSceneContext,
    useAnnotationScene,
} from '../../providers/annotation-scene-provider/annotation-scene-provider.component';
import {
    AnnotationToolProvider,
    useAnnotationToolContext,
} from '../../providers/annotation-tool-provider/annotation-tool-provider.component';
import { useROI } from '../../providers/region-of-interest-provider/region-of-interest-provider.component';
import { annotatorRender as render } from '../../test-utils/annotator-render';
import { SecondaryToolbar } from './secondary-toolbar.component';
import { SegmentAnythingStateProvider } from './segment-anything-state-provider.component';
import { SegmentAnythingTool } from './segment-anything-tool.component';
import { InteractiveAnnotationPoint } from './segment-anything.interface';
import { useSegmentAnythingModel } from './use-segment-anything-model.hook';

jest.mock('./use-segment-anything-model.hook', () => ({
    useSegmentAnythingModel: jest.fn(() => ({ encodingQuery: {}, modelIsReady: true, modelQuery: {} })),
}));

jest.mock('./../../zoom/zoom-provider.component', () => ({
    ...jest.requireActual('./../../zoom/zoom-provider.component'),
    useZoom: jest.fn(() => ({ zoomState: { zoom: 1.0, translation: { x: 0, y: 0 } } })),
}));

const NEW_ROI = { x: 0, y: 0, width: 100, height: 100 };

const fakeDecodingQueryFn = async (points: InteractiveAnnotationPoint[]): Promise<Shape[]> => {
    if (points.length === 0) {
        return [];
    }

    const x = Math.min(...points.map((point) => point.x));
    const y = Math.min(...points.map((point) => point.y));
    const x2 = Math.max(...points.map((point) => point.x));
    const y2 = Math.max(...points.map((point) => point.y));
    const width = x2 - x;
    const height = y2 - y;

    return [
        {
            shapeType: ShapeType.Rect,
            x: x - 50,
            y: y - 50,
            width: width + 100,
            height: height + 100,
        },
    ];
};

const mockROI = { x: 0, y: 0, width: 1000, height: 1000 };
const mockImage = getMockedImage(mockROI);

jest.mock('../../providers/region-of-interest-provider/region-of-interest-provider.component', () => ({
    ...jest.requireActual('../../providers/region-of-interest-provider/region-of-interest-provider.component'),
    useROI: jest.fn(() => ({
        roi: mockROI,
        image: mockImage,
    })),
}));

const expectNoPreviewShape = async () => {
    expect(screen.queryByLabelText('Segment anything preview')).not.toBeInTheDocument();
};

const expectPreviewShape = async ({ x, y, width, height }: Omit<Rect, 'shapeType'>) => {
    const preview = await screen.findByLabelText('Segment anything preview');
    expect(preview).toBeInTheDocument();

    const rect = preview.getElementsByTagName('rect').item(0);
    expect(rect).toHaveAttribute('x', `${x}`);
    expect(rect).toHaveAttribute('y', `${y}`);
    expect(rect).toHaveAttribute('width', `${width}`);
    expect(rect).toHaveAttribute('height', `${height}`);
};

const expectNoResultShape = async () => {
    expect(screen.queryByLabelText('Segment anything result')).not.toBeInTheDocument();
};

const expectResultShape = async ({ x, y, width, height }: Omit<Rect, 'shapeType'>) => {
    const preview = await screen.findByLabelText('Segment anything result');
    expect(preview).toBeInTheDocument();

    const rect = preview.getElementsByTagName('rect').item(0);
    expect(rect).toHaveAttribute('x', `${x}`);
    expect(rect).toHaveAttribute('y', `${y}`);
    expect(rect).toHaveAttribute('width', `${width}`);
    expect(rect).toHaveAttribute('height', `${height}`);
};

describe('Segment Anything', () => {
    const WithLabelShortcuts = () => {
        const annotationToolContext = useAnnotationToolContext();
        const {
            scene: { labels },
            toggleTool,
            tool,
        } = annotationToolContext;

        useEffect(() => {
            if (tool !== ToolType.SegmentAnythingTool) {
                toggleTool(ToolType.SegmentAnythingTool);
            }
        }, [toggleTool, tool]);

        return <LabelsShortcuts labels={[...labels]} annotationToolContext={annotationToolContext} />;
    };

    beforeEach(() => {
        // @ts-expect-error We don't care about the full value of encoding query
        jest.mocked(useSegmentAnythingModel).mockImplementation(() => ({
            encodingQuery: {
                isFetching: false,
                data: 'fake-data',
                remove: jest.fn(),
            },
            decodingQueryFn: fakeDecodingQueryFn,
        }));
    });

    afterAll(() => {
        jest.clearAllMocks();
    });

    const renderTool = async (withLabels = false) => {
        const addShapes = jest.fn();
        const WithAnnotationScene = ({ children }: { children: ReactNode }) => {
            const scene = useAnnotationScene();

            return (
                <AnnotationSceneContext.Provider value={{ ...scene, addShapes }}>
                    {children}
                </AnnotationSceneContext.Provider>
            );
        };

        const { container } = await render(
            <WithAnnotationScene>
                <AnnotationToolProvider>
                    <SegmentAnythingStateProvider>
                        <svg>
                            <SegmentAnythingTool />
                        </svg>
                        <SecondaryToolbar />
                        {withLabels && <WithLabelShortcuts />}
                    </SegmentAnythingStateProvider>
                </AnnotationToolProvider>
            </WithAnnotationScene>
        );

        return { addShapes, container };
    };

    it('previews an intermediate result on mouse move', async () => {
        const { addShapes } = await renderTool();

        const editor = await screen.findByRole('editor');
        fireEvent.pointerMove(editor, { clientX: 50, clientY: 50 });

        await expectPreviewShape({ x: 0, y: 0, width: 100, height: 100 });
        await expectNoResultShape();
        expect(addShapes).not.toHaveBeenCalled();

        // We no longer display the preview when the user is not focussed on the canvas
        fireEvent.pointerLeave(editor);
        await expectNoPreviewShape();
    });

    it('accepts a prediction from 1 single input', async () => {
        const { addShapes } = await renderTool();

        const editor = await screen.findByRole('editor');

        fireEvent.pointerMove(editor, { clientX: 150, clientY: 150 });
        await expectPreviewShape({ x: 100, y: 100, width: 100, height: 100 });

        fireEvent.pointerDown(editor, { buttons: 0, clientX: 150, clientY: 150 });
        fireEvent.pointerUp(editor, { buttons: 0, clientX: 150, clientY: 150 });

        await waitFor(() => {
            expect(addShapes).toHaveBeenCalled();
            expect(addShapes).toHaveBeenCalledWith([
                { shapeType: ShapeType.Rect, x: 100, y: 100, width: 100, height: 100 },
            ]);
        });
    });

    it('immediately accepts successive clicks', async () => {
        const { addShapes } = await renderTool();

        const editor = await screen.findByRole('editor');

        const points = [
            { x: 100, y: 100 },
            { x: 200, y: 100 },
            { x: 300, y: 100 },
        ];

        for (const { x, y } of points) {
            fireEvent.pointerMove(editor, { clientX: x, clientY: y });
            fireEvent.pointerDown(editor, { buttons: 0, clientX: x, clientY: y });
            fireEvent.pointerUp(editor, { buttons: 0, clientX: x, clientY: y });
        }

        await waitFor(() => {
            expect(addShapes).toHaveBeenCalledWith([
                { shapeType: ShapeType.Rect, x: 100 - 50, y: 100 - 50, width: 100, height: 100 },
            ]);

            expect(addShapes).toHaveBeenCalledWith([
                { shapeType: ShapeType.Rect, x: 200 - 50, y: 100 - 50, width: 100, height: 100 },
            ]);

            expect(addShapes).toHaveBeenCalledWith([
                { shapeType: ShapeType.Rect, x: 300 - 50, y: 100 - 50, width: 100, height: 100 },
            ]);
        });
    });

    describe('Interactive mode', () => {
        it('cancels an intermediate result', async () => {
            const { addShapes } = await renderTool();

            const editor = await screen.findByRole('editor');

            fireEvent.click(screen.getByRole('switch', { name: 'Interactive mode' }));

            fireEvent.pointerMove(editor, { clientX: 150, clientY: 150 });
            fireEvent.pointerDown(editor, { buttons: 0, clientX: 150, clientY: 150 });
            fireEvent.pointerUp(editor, { buttons: 0, clientX: 150, clientY: 150 });
            expect(screen.getAllByLabelText(/interactive segmentation point/)).toHaveLength(1);

            await waitFor(async () => {
                await expectNoPreviewShape();
                await expectResultShape({ x: 100, y: 100, width: 100, height: 100 });
            });

            fireEvent.click(screen.getByRole('button', { name: /reject/ }));
            expect(addShapes).not.toHaveBeenCalled();
            await expectNoResultShape();

            // Next accept it via keyboard shortcut
            fireEvent.pointerMove(editor, { clientX: 250, clientY: 250 });
            fireEvent.pointerDown(editor, { buttons: 0, clientX: 250, clientY: 250 });
            fireEvent.pointerUp(editor, { buttons: 0, clientX: 250, clientY: 250 });
            expect(screen.getAllByLabelText(/interactive segmentation point/)).toHaveLength(1);

            await waitFor(async () => {
                await expectNoPreviewShape();
                await expectResultShape({ x: 200, y: 200, width: 100, height: 100 });
            });

            await userEvent.keyboard('{escape}');

            await waitFor(async () => {
                await expectPreviewShape({ x: 200, y: 200, width: 100, height: 100 });
                await expectNoResultShape();
            });

            expect(addShapes).not.toHaveBeenCalled();
        });

        it('allows the user to correct an intermediate result when auto accept mode is turned off', async () => {
            const { addShapes } = await renderTool();

            const editor = await screen.findByRole('editor');
            fireEvent.click(screen.getByRole('switch', { name: 'Interactive mode' }));

            fireEvent.pointerMove(editor, { clientX: 150, clientY: 150 });
            fireEvent.pointerDown(editor, { buttons: 0, clientX: 150, clientY: 150 });
            fireEvent.pointerUp(editor, { buttons: 0, clientX: 150, clientY: 150 });

            expect(screen.getAllByLabelText(/positive interactive segmentation point/i)).toHaveLength(1);
            await expectNoPreviewShape();
            await expectResultShape({ x: 100, y: 100, width: 100, height: 100 });

            // Changing back to auto mode is disabled when the user has placed a point
            expect(screen.getByRole('switch', { name: 'Interactive mode' })).toBeDisabled();

            fireEvent.pointerMove(editor, { clientX: 250, clientY: 250 });
            await expectNoPreviewShape();
            await expectResultShape({ x: 100, y: 100, width: 100, height: 100 });

            fireEvent.pointerDown(editor, { buttons: 0, clientX: 250, clientY: 250 });
            fireEvent.pointerUp(editor, { buttons: 0, clientX: 250, clientY: 250 });
            expect(screen.getAllByLabelText(/positive interactive segmentation point/i)).toHaveLength(2);
            await expectResultShape({ x: 100, y: 100, width: 200, height: 200 });

            fireEvent.pointerDown(editor, { buttons: 0, clientX: 200, clientY: 200 });
            fireEvent.pointerUp(editor, { buttons: 0, clientX: 200, clientY: 200 });

            expect(screen.getAllByLabelText(/positive interactive segmentation point/i)).toHaveLength(2);
            expect(screen.getAllByLabelText(/negative interactive segmentation point/i)).toHaveLength(1);
            await expectResultShape({ x: 100, y: 100, width: 200, height: 200 });

            expect(addShapes).not.toHaveBeenCalled();
            fireEvent.click(screen.getByRole('button', { name: /accept/ }));
            expect(addShapes).toHaveBeenCalledWith([
                { shapeType: ShapeType.Rect, x: 100, y: 100, width: 200, height: 200 },
            ]);
        });

        it('accepts a result by clicking a label shortcut', async () => {
            const { addShapes } = await renderTool(true);

            const editor = await screen.findByRole('editor');
            fireEvent.click(screen.getByRole('switch', { name: 'Interactive mode' }));

            fireEvent.pointerMove(editor, { clientX: 150, clientY: 150 });
            fireEvent.pointerDown(editor, { buttons: 0, clientX: 150, clientY: 150 });
            fireEvent.pointerUp(editor, { buttons: 0, clientX: 150, clientY: 150 });

            await expectResultShape({ x: 100, y: 100, width: 100, height: 100 });

            const cardShortcut = screen.getByRole('button', { name: 'card' });
            fireEvent.click(cardShortcut);

            await expectNoResultShape();
            expect(addShapes).toHaveBeenCalled();
            expect(addShapes).toHaveBeenCalledWith([
                { shapeType: ShapeType.Rect, x: 100, y: 100, width: 100, height: 100 },
            ]);
        });

        it('accepts a result by clicking a label keyboard shortcut', async () => {
            const { addShapes } = await renderTool(true);

            const editor = await screen.findByRole('editor');
            fireEvent.click(screen.getByRole('switch', { name: 'Interactive mode' }));

            fireEvent.pointerMove(editor, { clientX: 150, clientY: 150 });
            fireEvent.pointerDown(editor, { buttons: 0, clientX: 150, clientY: 150 });
            fireEvent.pointerUp(editor, { buttons: 0, clientX: 150, clientY: 150 });

            await expectResultShape({ x: 100, y: 100, width: 100, height: 100 });

            await userEvent.keyboard('{Control>}1{/Control}');

            await expectNoResultShape();
            expect(addShapes).toHaveBeenCalled();
            expect(addShapes).toHaveBeenCalledWith([
                { shapeType: ShapeType.Rect, x: 100, y: 100, width: 100, height: 100 },
            ]);
        });

        describe('using right click mode', () => {
            it('does not add negative point when right click mode is disabled', async () => {
                await renderTool();

                const editor = await screen.findByRole('editor');
                fireEvent.click(screen.getByRole('switch', { name: 'Interactive mode' }));

                fireEvent.pointerMove(editor, { clientX: 150, clientY: 150 });
                fireEvent.pointerDown(editor, { buttons: 0, clientX: 150, clientY: 150 });
                fireEvent.pointerUp(editor, { buttons: 0, clientX: 150, clientY: 150 });

                fireEvent.pointerMove(editor, { clientX: 120, clientY: 120 });
                fireEvent.pointerDown(editor, { buttons: 2, clientX: 120, clientY: 10 });
                fireEvent.pointerUp(editor, { buttons: 2, clientX: 120, clientY: 120 });

                expect(screen.getByRole('switch', { name: 'Right-click mode' })).not.toBeChecked();
                expect(screen.queryByLabelText('Negative interactive segmentation point')).not.toBeInTheDocument();

                // Prevent error about setIsDrawing being called after test
                fireEvent.click(screen.getByRole('button', { name: /reject/ }));
            });

            it('places a positive point by left clicking and negative point by right clicking', async () => {
                await renderTool();

                const editor = await screen.findByRole('editor');
                fireEvent.click(screen.getByRole('switch', { name: 'Interactive mode' }));
                fireEvent.click(screen.getByRole('switch', { name: 'Right-click mode' }));

                fireEvent.pointerMove(editor, { clientX: 150, clientY: 150 });
                fireEvent.pointerDown(editor, { buttons: 0, clientX: 150, clientY: 150 });
                fireEvent.pointerUp(editor, { buttons: 0, clientX: 150, clientY: 150 });

                await expectNoPreviewShape();
                await expectResultShape({ x: 100, y: 100, width: 100, height: 100 });

                fireEvent.pointerMove(editor, { clientX: 100, clientY: 150 });
                fireEvent.pointerDown(editor, { buttons: 0, clientX: 100, clientY: 150 });
                fireEvent.pointerUp(editor, { buttons: 0, clientX: 100, clientY: 150 });

                expect(screen.getAllByLabelText(/positive interactive segmentation point/i)).toHaveLength(2);
                expect(screen.queryAllByLabelText(/negative interactive segmentation point/i)).toHaveLength(0);

                await expectResultShape({ x: 50, y: 100, width: 150, height: 100 });
                await expectNoPreviewShape();

                fireEvent.pointerMove(editor, { clientX: 300, clientY: 150 });
                fireEvent.pointerDown(editor, { buttons: 2, clientX: 300, clientY: 150 });
                fireEvent.pointerUp(editor, { buttons: 2, clientX: 300, clientY: 150 });

                expect(screen.getAllByLabelText(/positive interactive segmentation point/i)).toHaveLength(2);
                expect(screen.queryAllByLabelText(/negative interactive segmentation point/i)).toHaveLength(1);
            });

            it('does not place a negative point if there are no positive points', async () => {
                await renderTool();

                const editor = await screen.findByRole('editor');
                fireEvent.click(screen.getByRole('switch', { name: 'Interactive mode' }));
                fireEvent.click(screen.getByRole('switch', { name: 'Right-click mode' }));

                fireEvent.pointerMove(editor, { clientX: 150, clientY: 150 });
                fireEvent.pointerDown(editor, { buttons: 2, clientX: 150, clientY: 150 });
                fireEvent.pointerUp(editor, { buttons: 2, clientX: 150, clientY: 150 });

                expect(screen.queryAllByLabelText(/positive interactive segmentation point/i)).toHaveLength(0);
                expect(screen.queryAllByLabelText(/negative interactive segmentation point/i)).toHaveLength(0);

                // We do show a preview had the user used left click
                await expectPreviewShape({ x: 100, y: 100, width: 100, height: 100 });
            });
        });
    });

    it('allows the user to change the opacity of the mask used to obfuscate the image', async () => {
        await renderTool();

        const editor = await screen.findByRole('editor');

        fireEvent.pointerMove(editor, { clientX: 150, clientY: 150 });

        await expectPreviewShape({ x: 100, y: 100, width: 100, height: 100 });

        const mask = screen.getByLabelText('Annotation mask');
        const rect = mask.getElementsByTagName('rect').item(0);
        expect(rect).toHaveAttribute('fill-opacity', '0.3');

        // Change the opacity
        fireEvent.click(screen.getByRole('button', { name: /Mask opacity/i }));
        const slider = await screen.findByRole('slider', { name: 'Mask opacity slider' });
        fireEvent.change(slider, { target: { value: 0.89 } });
        fireEvent.keyDown(slider, { key: 'Right' });

        expect(screen.getByText('90%')).toBeInTheDocument();
        expect(rect).toHaveAttribute('fill-opacity', '0.9');
    });

    it('allows cancelling when there are multiple points but no shapes', async () => {
        // @ts-expect-error We don't care about the full value of encoding query
        jest.mocked(useSegmentAnythingModel).mockImplementation(() => ({
            encodingQuery: {
                isFetching: false,
                data: 'fake-data',
                remove: jest.fn(),
            },
            decodingQueryFn: async () => [],
        }));

        await renderTool();

        const editor = await screen.findByRole('editor');
        fireEvent.click(screen.getByRole('switch', { name: 'Interactive mode' }));
        fireEvent.click(screen.getByRole('switch', { name: 'Right-click mode' }));

        fireEvent.pointerMove(editor, { clientX: 150, clientY: 150 });
        fireEvent.pointerDown(editor, { buttons: 0, clientX: 150, clientY: 150 });
        fireEvent.pointerUp(editor, { buttons: 0, clientX: 150, clientY: 150 });

        fireEvent.pointerMove(editor, { clientX: 250, clientY: 250 });
        fireEvent.pointerDown(editor, { buttons: 0, clientX: 250, clientY: 250 });
        fireEvent.pointerUp(editor, { buttons: 0, clientX: 250, clientY: 250 });

        await expectNoPreviewShape();
        await expectNoResultShape();

        expect(screen.getAllByLabelText(/positive interactive segmentation point/i)).toHaveLength(2);

        expect(screen.getByRole('button', { name: /accept/ })).toBeDisabled();
        expect(screen.getByRole('button', { name: /reject/ })).toBeEnabled();

        await waitFor(async () => {
            fireEvent.click(screen.getByRole('button', { name: /reject/ }));

            expect(screen.queryAllByLabelText(/positive interactive segmentation point/i)).toHaveLength(0);
        });
    });

    it('shows a loading indicator while extracting image features', async () => {
        // @ts-expect-error We don't care about the full value of encoding query
        jest.mocked(useSegmentAnythingModel).mockImplementation(() => ({
            encodingQuery: {
                isFetching: true,
                data: undefined,
                remove: jest.fn(),
            },
            decodingQueryFn: fakeDecodingQueryFn,
        }));

        await renderTool();

        expect(await screen.findByText('Extracting image features')).toBeInTheDocument();
    });

    describe('task chain', () => {
        const TASK_CHAIN_ROI = { x: 50, y: 50, width: 100, height: 100 };

        beforeEach(() => {
            jest.mocked(useROI).mockImplementation(() => ({
                roi: TASK_CHAIN_ROI,
                image: mockImage,
            }));
        });

        it('does not let the user place points outside of the roi', async () => {
            await renderTool();

            const editor = await screen.findByRole('editor');

            fireEvent.pointerMove(editor, { clientX: 20, clientY: 20 });
            fireEvent.pointerDown(editor, { buttons: 0, clientX: 20, clientY: 20 });
            fireEvent.pointerUp(editor, { buttons: 0, clientX: 20, clientY: 20 });
            expect(screen.queryAllByLabelText(/interactive segmentation point/)).toHaveLength(0);

            await expectNoPreviewShape();
            await expectNoResultShape();
        });

        it('removes points from the model output that are outisde of the roi', async () => {
            const { addShapes } = await renderTool();

            const editor = await screen.findByRole('editor');

            fireEvent.pointerMove(editor, { clientX: 75, clientY: 75 });
            fireEvent.pointerDown(editor, { buttons: 0, clientX: 75, clientY: 75 });
            fireEvent.pointerUp(editor, { buttons: 0, clientX: 75, clientY: 75 });

            await waitFor(() => {
                expect(addShapes).toHaveBeenCalledWith([
                    { shapeType: ShapeType.Rect, x: 50, y: 50, width: 75, height: 75 },
                ]);
            });
        });

        it('updates the roi', async () => {
            await renderTool();

            const editor = await screen.findByRole('editor');

            fireEvent.pointerMove(editor, { clientX: 75, clientY: 75 });
            await expectPreviewShape({ x: 50, y: 50, width: 75, height: 75 });

            jest.mocked(useROI).mockImplementation(() => ({
                roi: NEW_ROI,
                image: mockImage,
            }));

            // Moving up forces a rerender which makes the ROI change
            fireEvent.pointerDown(editor, { buttons: 0, clientX: 75, clientY: 75 });
            fireEvent.pointerUp(editor, { buttons: 0, clientX: 75, clientY: 75 });

            fireEvent.pointerMove(editor, { clientX: 75, clientY: 75 });
            await waitFor(async () => {
                await expectPreviewShape({ x: 25, y: 25, width: 75, height: 75 });
            });
        });
    });
});
