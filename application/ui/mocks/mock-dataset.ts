// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { components } from '../src/api/openapi-spec';

type MediaView = components['schemas']['MediaView'];

export const mockedMedia = (props: Partial<MediaView> = {}): MediaView => ({
    id: 'item-1',
    type: 'image',
    name: 'item-1.jpg',
    format: 'jpg',
    width: 0,
    height: 0,
    size: 0,
    ...props,
});
