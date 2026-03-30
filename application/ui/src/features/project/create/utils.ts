// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const generateUniqueProjectName = (existingNames: string[]): string => {
    const usedNumbers: number[] = [];

    existingNames.forEach((name) => {
        const match = name.match(/^Project #(\d+)$/);
        if (match) {
            usedNumbers.push(Number(match[1]));
        }
    });

    if (usedNumbers.length === 0) {
        return 'Project #1';
    }

    const maxNumber = Math.max(...usedNumbers);
    return `Project #${maxNumber + 1}`;
};
