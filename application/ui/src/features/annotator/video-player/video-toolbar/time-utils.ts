// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

const getTimeUnits = (duration: number) => {
    const hours = Math.floor(duration / (60 * 60));
    const hoursSeconds = hours * 60 * 60;

    const minutes = Math.floor((duration - hoursSeconds) / 60);
    const minutesSeconds = 60 * minutes;

    const seconds = duration - hoursSeconds - minutesSeconds;

    return { hours, minutes, seconds };
};

const paddedString = (number: number): string => Math.floor(number).toString().padStart(2, '0');

export const useDurationText = (duration: number): string => {
    const { hours, minutes, seconds } = getTimeUnits(duration);

    return `${paddedString(hours)}:${paddedString(minutes)}:${paddedString(seconds)}`;
};
