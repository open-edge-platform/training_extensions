// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest';

import { mapProjectLabels } from './util';

describe('util - label mapping', () => {
    it('should return an empty mapping when datasetLabels is empty', () => {
        const formData = new FormData();
        const mapping = mapProjectLabels([], formData);

        expect(mapping).toEqual({});
    });

    it('should map labels when corresponding target labels are provided', () => {
        const datasetLabels = ['cat', 'dog', 'bird'];
        const formData = new FormData();
        formData.set('targetLabel-0', 'feline');
        formData.set('targetLabel-1', 'canine');
        formData.set('targetLabel-2', 'avian');

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({
            cat: 'feline',
            dog: 'canine',
            bird: 'avian',
        });
    });

    it('should skip labels when the target label is missing', () => {
        const datasetLabels = ['cat', 'dog'];
        const formData = new FormData();
        formData.set('targetLabel-0', 'feline');
        // targetLabel-1 is intentionally not set

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({
            cat: 'feline',
        });
        expect(mapping).not.toHaveProperty('dog');
    });

    it('should skip labels when the target label is an empty string', () => {
        const datasetLabels = ['cat', 'dog'];
        const formData = new FormData();
        formData.set('targetLabel-0', '');
        formData.set('targetLabel-1', 'canine');

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({
            dog: 'canine',
        });
        expect(mapping).not.toHaveProperty('cat');
    });

    it('should skip labels when the target label is a File (non-string)', () => {
        const datasetLabels = ['image'];
        const formData = new FormData();
        const file = new File(['content'], 'image.txt', { type: 'text/plain' });
        formData.set('targetLabel-0', file);

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({});
    });

    it('should handle sparse mappings correctly', () => {
        const datasetLabels = ['cat', 'dog', 'bird', 'fish'];
        const formData = new FormData();
        formData.set('targetLabel-0', 'feline');
        // skip index 1
        formData.set('targetLabel-2', 'avian');
        // skip index 3

        const mapping = mapProjectLabels(datasetLabels, formData);

        expect(mapping).toEqual({
            cat: 'feline',
            bird: 'avian',
        });
        expect(mapping).not.toHaveProperty('dog');
        expect(mapping).not.toHaveProperty('fish');
    });
});
