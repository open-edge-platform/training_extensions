// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DatasetSubset } from '../../../constants/shared-types';
import { useAnnotationActions } from '../../../shared/annotator/annotation-actions-provider.component';

type UseSubmitAnnotationsProps = {
    onSuccess?: () => void;
    subset?: DatasetSubset;
};
export const useSubmitAnnotations = ({ onSuccess, subset }: UseSubmitAnnotationsProps) => {
    const { canSubmit, isSaving, submitAnnotations } = useAnnotationActions();

    const handleSubmitAnnotations = async () => {
        await submitAnnotations(subset);
        onSuccess?.();
    };

    return { canSubmit, isSaving, submit: handleSubmitAnnotations };
};
