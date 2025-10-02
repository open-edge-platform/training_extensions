// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { test as base, expect } from '@playwright/test';

// E2E tests should NOT use MSW mocking - they should hit the real backend
// This is a clean fixtures file without any network mocking

export const test = base;

export { expect };
