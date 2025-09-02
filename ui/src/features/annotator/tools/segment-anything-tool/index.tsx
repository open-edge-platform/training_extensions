// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SegmentAnythingIcon } from '@geti/ui/icons';

import { SecondaryToolbar } from './secondary-toolbar.component';
import { SegmentAnythingStateProvider } from './segment-anything-state-provider.component';
import { SegmentAnythingTool as Tool } from './segment-anything-tool.component';

export const SegmentAnythingTool = {
    type: 'segment-anything-tool',
    label: 'Auto segmentation',
    Icon: () => <SegmentAnythingIcon />,
    Tool,
    SecondaryToolbar,
    StateProvider: SegmentAnythingStateProvider,
};
