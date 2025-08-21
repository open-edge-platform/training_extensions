// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export type WizardStep = {
    href: string;
    name: string;
    isDisabled: boolean;
    isCompleted: boolean;
    isSelected: boolean;
};

export type WizardState = Array<WizardStep>;
