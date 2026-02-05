// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import prettyBytes from 'pretty-bytes';

export const formatModelSize = (bytes: number): string => prettyBytes(bytes);
