// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const VALID_VIDEO_EXT = ['mp4', 'avi', 'mkv', 'mov', 'webm', 'm4v'];
export const VALID_IMAGE_EXT = ['jpg', 'jpeg', 'png', 'jfif', 'tif', 'tiff', 'webp', 'bmp'];
export const VALID_EXT = [...VALID_VIDEO_EXT, ...VALID_IMAGE_EXT];

const getFileExtension = (file: File): string => file.name.split('.').pop()?.toLowerCase() ?? '';

export const isVideoFile = (file: File) => VALID_VIDEO_EXT.includes(getFileExtension(file));

export const isSupportedMediaFile = (file: File): boolean => VALID_EXT.includes(getFileExtension(file));
