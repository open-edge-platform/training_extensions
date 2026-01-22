// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { components } from '../src/api/openapi-spec';

type DatasetItemView = components['schemas']['DatasetItemView'];

export const mockedDatasetItem = (props: Partial<DatasetItemView> = {}): DatasetItemView => ({
    id: 'item-1',
    name: 'item-1.jpg',
    format: 'jpg',
    width: 0,
    height: 0,
    size: 0,
    subset: 'unassigned',
    ...props,
});
