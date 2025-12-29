// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label } from '../../../constants/shared-types';

export type LabelItemProps = {
    label: Label;
    onDelete: (id: string) => void;
    onUpdate: (label: Label) => void;
};
