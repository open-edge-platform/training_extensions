// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Label } from '../../annotator/types';

export type LabelItemProps = {
    label: Label;
    onDelete: (id: string) => void;
    onUpdate: (label: Label) => void;
};
