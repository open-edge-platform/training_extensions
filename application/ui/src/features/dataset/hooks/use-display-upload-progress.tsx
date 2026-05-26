// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { v4 as uuid } from 'uuid';

import { useMediaUploadContext } from '../providers/media-upload-provider.component';
import { computeSummary, UploadFileItem, UploadProgressSummary } from '../providers/media-upload-reducer';

type UseUploadProgressApi = {
    uploadProgress: UploadProgressSummary;
    startUploadProgress: (files: File[]) => string[];
    setItemUploading: (itemId: string) => void;
    setItemUploaded: (itemId: string) => void;
    setItemFailed: (itemId: string, errorMessage?: string) => void;
    finishUploadProgress: () => void;
};

export const useUploadProgress = (): UseUploadProgressApi => {
    const { state, dispatch } = useMediaUploadContext();
    const uploadProgress = useMemo(
        () => computeSummary(state.items, state.isUploading),
        [state.items, state.isUploading]
    );

    const startUploadProgress = (files: File[]): string[] => {
        const newItems: UploadFileItem[] = files.map((file) => ({
            id: uuid(),
            name: file.name,
            size: file.size,
            status: 'queued',
        }));

        dispatch({ type: 'START_UPLOAD', payload: newItems });

        return newItems.map((item) => item.id);
    };

    const setItemUploading = (itemId: string): void => {
        dispatch({ type: 'SET_UPLOADING', payload: { itemId } });
    };

    const setItemUploaded = (itemId: string): void => {
        dispatch({ type: 'SET_UPLOADED', payload: { itemId } });
    };

    const setItemFailed = (itemId: string, errorMessage?: string): void => {
        dispatch({ type: 'SET_FAILED', payload: { itemId, errorMessage } });
    };

    const finishUploadProgress = (): void => {
        dispatch({ type: 'FINISH_UPLOAD' });
    };

    return {
        uploadProgress,
        startUploadProgress,
        setItemUploading,
        setItemUploaded,
        setItemFailed,
        finishUploadProgress,
    };
};
