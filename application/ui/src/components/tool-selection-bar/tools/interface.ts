// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ComponentType, SVGProps } from 'react';

export type ToolType = 'selection' | 'bounding-box' | 'polygon' | 'sam';

export interface ToolConfig {
    type: ToolType;
    icon: ComponentType<SVGProps<SVGSVGElement>>;
}
