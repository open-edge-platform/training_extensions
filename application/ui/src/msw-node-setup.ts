// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { setupServer } from 'msw/node';

import { handlers } from './api/utils';

// Initialize msw's mock server with the handlers
export const server = setupServer(...handlers);
