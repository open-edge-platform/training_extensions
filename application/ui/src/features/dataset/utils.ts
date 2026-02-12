// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Media } from '../../constants/shared-types';

export const isVideo = (
    media: Partial<Pick<Media, 'type' | 'frame_count'>> | undefined
): media is Pick<Media, 'type'> & { frame_count: number } => media?.type === 'video' && media.frame_count != null;
