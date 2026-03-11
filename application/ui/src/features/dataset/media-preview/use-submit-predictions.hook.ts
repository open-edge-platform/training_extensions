// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';

export const useSubmitPredictions = ({ onSuccess }: { onSuccess?: () => void }) => {
    const { canSubmit, isSaving, submitAnnotations } = useAnnotationActions();

    const handleSubmitPredictions = async () => {
        await submitAnnotations();
        onSuccess?.();
    };

    return { canSubmit, isSaving, submit: handleSubmitPredictions };
};
