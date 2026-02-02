// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const validateProjectName = (name: string, projectNames: string[]): string | undefined => {
    if (name.trim().length === 0) {
        return 'Project name cannot be empty';
    }

    if (projectNames.includes(name)) {
        return 'That project name already exists';
    }

    return undefined;
};
