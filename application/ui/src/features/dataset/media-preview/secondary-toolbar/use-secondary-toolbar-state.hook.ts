// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useSelectedAnnotations } from '../../../../shared/annotator/select-annotation-provider.component';

export const useSecondaryToolbarState = () => {
    const { selectedAnnotations } = useSelectedAnnotations();

    const isHidden = selectedAnnotations.size === 0;

    return {
        isHidden,
    };
};
