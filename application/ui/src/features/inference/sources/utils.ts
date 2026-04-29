// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const getUniqueName = (baseName: string, existingNames: string[]): string => {
    const existingNameSet = new Set(existingNames);

    if (!existingNameSet.has(baseName)) {
        return baseName;
    }

    let counter = 1;
    while (existingNameSet.has(`${baseName} - ${counter}`)) {
        counter++;
    }
    return `${baseName} - ${counter}`;
};
