// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { useProject } from 'hooks/api/project.hook';

import { isClassificationTask, isMultiLabelClassificationTask } from '../../../project/task-type-guards';
import { useMediaUpload } from '../../api/use-media-upload';

export const useBulkUploadAndAssignLabel = () => {
    const { data: project } = useProject();
    const { uploadMedia, uploadProgress } = useMediaUpload();

    const [filesForLabelAssignment, setFilesForLabelAssignment] = useState<File[]>([]);

    const isClassification = isClassificationTask(project.task.task_type);
    const isMultiLabelClassification = isMultiLabelClassificationTask(project.task);

    const handleFileUpload = async (files: File[]) => {
        if (isClassification) {
            setFilesForLabelAssignment(files);
        } else {
            await uploadMedia(files);
        }
    };

    return {
        uploadMedia,
        uploadMediaLoading: uploadProgress.isUploading,
        uploadAndAssign: handleFileUpload,
        isClassification,
        isMultiLabelClassification,

        filesForLabelAssignment,
        setFilesForLabelAssignment,
    };
};
