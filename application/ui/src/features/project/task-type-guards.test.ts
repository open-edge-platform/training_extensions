// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { getMockedProject } from 'mocks/mock-project';

import { TaskType } from '../../constants/shared-types';
import {
    isClassificationTask,
    isDetectionTask,
    isMultiLabelClassificationTask,
    isPrefetchEnabledForTask,
    isSegmentationTask,
} from './task-type-guards';

const ALL_TASK_TYPES: TaskType[] = ['classification', 'detection', 'instance_segmentation'];

describe('isClassificationTask', () => {
    it('returns true for "classification"', () => {
        expect(isClassificationTask('classification')).toBe(true);
    });

    it.each(['detection', 'instance_segmentation'] satisfies TaskType[])('returns false for "%s"', (taskType) => {
        expect(isClassificationTask(taskType)).toBe(false);
    });

    it('returns false for null', () => {
        expect(isClassificationTask(null)).toBe(false);
    });
});

describe('isDetectionTask', () => {
    it('returns true for "detection"', () => {
        expect(isDetectionTask('detection')).toBe(true);
    });

    it.each(['classification', 'instance_segmentation'] satisfies TaskType[])('returns false for "%s"', (taskType) => {
        expect(isDetectionTask(taskType)).toBe(false);
    });

    it('returns false for null', () => {
        expect(isDetectionTask(null)).toBe(false);
    });
});

describe('isSegmentationTask', () => {
    it('returns true for "instance_segmentation"', () => {
        expect(isSegmentationTask('instance_segmentation')).toBe(true);
    });

    it.each(['classification', 'detection'] satisfies TaskType[])('returns false for "%s"', (taskType) => {
        expect(isSegmentationTask(taskType)).toBe(false);
    });

    it('returns false for null', () => {
        expect(isSegmentationTask(null)).toBe(false);
    });
});

describe('isPrefetchEnabledForTask', () => {
    it('returns true for "detection"', () => {
        expect(isPrefetchEnabledForTask('detection')).toBe(true);
    });

    it('returns true for "instance_segmentation"', () => {
        expect(isPrefetchEnabledForTask('instance_segmentation')).toBe(true);
    });

    it('returns false for "classification"', () => {
        expect(isPrefetchEnabledForTask('classification')).toBe(false);
    });

    it('returns false for null', () => {
        expect(isPrefetchEnabledForTask(null)).toBe(false);
    });

    it('returns true only for detection and instance_segmentation among all task types', () => {
        const enabledTypes = ALL_TASK_TYPES.filter((t) => isPrefetchEnabledForTask(t));
        expect(enabledTypes).toEqual(expect.arrayContaining(['detection', 'instance_segmentation']));
        expect(enabledTypes).not.toContain('classification');
    });
});

describe('isMultiLabelClassificationTask', () => {
    it('returns true when task is classification and exclusive_labels is false', () => {
        const task = getMockedProject({
            task: { task_type: 'classification', exclusive_labels: false, labels: [] },
        }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(true);
    });

    it('returns false when task is classification but exclusive_labels is true', () => {
        const task = getMockedProject({
            task: { task_type: 'classification', exclusive_labels: true, labels: [] },
        }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(false);
    });

    it('returns false when task is detection and exclusive_labels is false', () => {
        const task = getMockedProject({ task: { task_type: 'detection', exclusive_labels: false, labels: [] } }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(false);
    });

    it('returns false when task is detection and exclusive_labels is true', () => {
        const task = getMockedProject({ task: { task_type: 'detection', exclusive_labels: true, labels: [] } }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(false);
    });

    it('returns false when task is instance_segmentation and exclusive_labels is false', () => {
        const task = getMockedProject({
            task: { task_type: 'instance_segmentation', exclusive_labels: false, labels: [] },
        }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(false);
    });

    it('returns false when task is instance_segmentation and exclusive_labels is true', () => {
        const task = getMockedProject({
            task: { task_type: 'instance_segmentation', exclusive_labels: true, labels: [] },
        }).task;
        expect(isMultiLabelClassificationTask(task)).toBe(false);
    });
});
