// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export type ImportDatasetToProjectState = 'uploading' | 'preparing' | 'labelMapping' | 'importing';
export type ImportDatasetAsNewProjectState = ImportDatasetToProjectState | 'taskTypeSelection';
