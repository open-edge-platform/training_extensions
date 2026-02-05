// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { isEmpty } from 'lodash-es';

import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';

export const useSubmitPredictions = ({ onSuccess }: { onSuccess?: () => void }) => {
    const { annotations, isSaving, submitAnnotations } = useAnnotationActions();

    const handleSubmitPredictions = async () => {
        await submitAnnotations();
        onSuccess?.();
    };

    return { canSubmit: !isEmpty(annotations), isSaving, submit: handleSubmitPredictions };
};
