// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const getRandomDistinctColor = (): string => DISTINCT_COLORS[Math.floor(Math.random() * DISTINCT_COLORS.length)];

export const DISTINCT_COLORS = [
    '#708541',
    '#E96115',
    '#EDB200',
    '#FF5662',
    '#CC94DA',
    '#5B69FF',
    '#548FAD',
    //
    '#25A18E',
    '#9D3B1A',
    '#C9E649',
    '#F15B85',
    '#81407B',
    '#26518E',
    '#076984',
    //
    '#00F5D4',
    '#FF7D00',
    '#F7DAB3',
    '#80E9AF',
    '#9B5DE5',
    '#00A5CF',
    '#D7BC5E',
];
