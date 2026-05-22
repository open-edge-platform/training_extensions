// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TrainingDevice } from '../../../../constants/shared-types';

const GPU_DEVICE_TYPES = ['xpu', 'cuda'];

/**
 * Selects the best default training device from the available devices.
 *
 * Priority:
 * 1. GPU devices (xpu or cuda) are preferred over CPU
 * 2. Among GPU devices, the one with the highest memory is chosen
 * 3. If memory is equal, the one with the lowest index is chosen
 * 4. Falls back to the first available device (typically CPU) if no GPU is found
 */
export const getDefaultTrainingDevice = (trainingDevices: TrainingDevice[]): TrainingDevice | undefined => {
    const gpuDevices = trainingDevices.filter((device) => GPU_DEVICE_TYPES.includes(device.type));

    if (gpuDevices.length === 0) {
        return trainingDevices.at(0);
    }

    return gpuDevices.reduce((best, current) => {
        const bestMemory = best.memory ?? -Infinity;
        const currentMemory = current.memory ?? -Infinity;

        if (currentMemory > bestMemory) {
            return current;
        }

        if (currentMemory === bestMemory) {
            const bestIndex = best.index ?? Infinity;
            const currentIndex = current.index ?? Infinity;

            if (currentIndex < bestIndex) {
                return current;
            }
        }

        return best;
    });
};

const formatDeviceMemory = (bytes: number): string => {
    return `${Math.ceil(bytes / 1024 ** 3)} GB`;
};

export const createDeviceName = (device: TrainingDevice): string => {
    let name = device.name;

    if (device.memory != null) {
        const memory = formatDeviceMemory(device.memory);
        name += ` (${memory})`;
    }

    if (device.index != null) {
        name += ` [${device.index}]`;
    }

    return name;
};
