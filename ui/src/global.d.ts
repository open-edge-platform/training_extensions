// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/// <reference types="@rsbuild/core/types" />

import 'react';

import { type NavigateOptions } from 'react-router-dom';

declare module '@adobe/react-spectrum' {
    interface RouterConfig {
        routerOptions: NavigateOptions;
    }
}

declare module 'react' {
    interface CSSProperties {
        [key: `--${string}`]: string | number;
    }
}
