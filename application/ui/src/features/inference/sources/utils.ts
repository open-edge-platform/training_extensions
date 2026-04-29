// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const getUniqueName = (baseName: string, existingNames: string[]): string => {
    if (!existingNames.includes(baseName)) {
        return baseName;
    }

    let counter = 1;
    while (existingNames.includes(`${baseName} - ${counter}`)) {
        counter++;
    }
    return `${baseName} - ${counter}`;
};
