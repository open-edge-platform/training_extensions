// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TrainingConfiguration } from '../../../../constants/shared-types';
import { mockedTrainingConfiguration } from './mocks';
import { getTrainingConfigurationUpdatePayload } from './utils';

describe('getTrainingConfigurationUpdatePayload', () => {
    it('returns empty object when config is undefined', () => {
        expect(getTrainingConfigurationUpdatePayload(undefined)).toEqual({});
    });

    it('returns empty object when config has no parameters', () => {
        const emptyConfig: TrainingConfiguration = { parameters: [] };
        expect(getTrainingConfigurationUpdatePayload(emptyConfig)).toEqual({});
    });

    it('converts the full mockedTrainingConfiguration to the correct update payload', () => {
        expect(getTrainingConfigurationUpdatePayload(mockedTrainingConfiguration)).toEqual({
            'dataset_preparation.subset_split.training': 70,
            'dataset_preparation.subset_split.validation': 20,
            'dataset_preparation.subset_split.test': 10,
            'dataset_preparation.filtering.min_annotation_pixels.enable': false,
            'dataset_preparation.filtering.min_annotation_pixels.value': 1,
            'dataset_preparation.filtering.min_annotation_objects.enable': false,
            'dataset_preparation.filtering.min_annotation_objects.value': 1,
            'dataset_preparation.filtering.max_annotation_objects.enable': false,
            'dataset_preparation.filtering.max_annotation_objects.value': 10000,
            'dataset_preparation.augmentation.random_affine.enable': false,
            'dataset_preparation.augmentation.random_affine.max_rotate_degree': 10,
            'dataset_preparation.augmentation.random_affine.max_translate_ratio': 0.1,
            'dataset_preparation.augmentation.random_affine.scaling_ratio_range': [0.5, 1.5],
            'dataset_preparation.augmentation.random_affine.max_shear_degree': 2,
            'dataset_preparation.augmentation.random_horizontal_flip.enable': true,
            'dataset_preparation.augmentation.random_horizontal_flip.probability': 0.5,
            'dataset_preparation.augmentation.random_vertical_flip.enable': false,
            'dataset_preparation.augmentation.random_vertical_flip.probability': 0.5,
            'dataset_preparation.augmentation.color_jitter.enable': false,
            'dataset_preparation.augmentation.color_jitter.brightness': [0.875, 1.125],
            'dataset_preparation.augmentation.color_jitter.contrast': [0.5, 1.5],
            'dataset_preparation.augmentation.color_jitter.saturation': [0.5, 1.5],
            'dataset_preparation.augmentation.color_jitter.hue': [-0.05, 0.05],
            'dataset_preparation.augmentation.color_jitter.probability': 0.5,
            'dataset_preparation.augmentation.gaussian_blur.enable': false,
            'dataset_preparation.augmentation.gaussian_blur.kernel_size': 5,
            'dataset_preparation.augmentation.gaussian_blur.sigma': [0.1, 2],
            'dataset_preparation.augmentation.gaussian_blur.probability': 0.5,
            'dataset_preparation.augmentation.gaussian_noise.enable': false,
            'dataset_preparation.augmentation.gaussian_noise.mean': 0,
            'dataset_preparation.augmentation.gaussian_noise.sigma': 0.1,
            'dataset_preparation.augmentation.gaussian_noise.probability': 0.5,
            'dataset_preparation.augmentation.tiling.enable': false,
            'dataset_preparation.augmentation.tiling.enable_adaptive_tiling': true,
            'dataset_preparation.augmentation.tiling.tile_size': 400,
            'dataset_preparation.augmentation.tiling.tile_overlap': 0.2,
            'training.max_epochs': 200,
            'training.batch_size': 4,
            'training.early_stopping.enable': true,
            'training.early_stopping.patience': 10,
            'training.learning_rate': 0.007,
            'training.weight_decay': 0.0001,
            'training.scheduler.type': 'reduce_lr_on_plateau',
            'training.scheduler.warmup.enable': false,
            'training.scheduler.warmup.epochs': 5,
            'training.scheduler.factor': 0.5,
            'training.scheduler.patience': 5,
            'training.scheduler.min_lr': 0,
            'training.gradient_accumulation.enable': false,
            'training.gradient_accumulation.batches': 1,
            'training.gradient_clip.enable': false,
            'training.gradient_clip.max_grad_norm': 1,
            'training.input_size_width': 1024,
            'training.input_size_height': 1024,
            'evaluation.validation_metric': 'default',
        });
    });
});
