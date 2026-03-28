// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const VALID_DATASET_TYPES = ['zip'];

export const formatToFileArray = (files: FileList | File[] | null): File[] => {
    return (files && Array.from(files)) ?? [];
};

export const isSupportedDatasetZip = (file: File): boolean => {
    const fileType = file.type ? file.type.split('/')[1] : '';

    return fileType ? VALID_DATASET_TYPES.some((type) => fileType.includes(type)) : false;
};
