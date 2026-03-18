// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    getMockedConfigurationParameter,
    getMockedConfigurationParameterGroup,
} from 'mocks/mock-training-configuration';

import {
    ConfigurableParameter,
    ConfigurableParameterGroup,
    TrainingConfigurationParameter,
} from '../../../../constants/shared-types';
import { isParameterGroup } from '../../model-listing/model-training-parameters/utils';
import { learningParameters } from './training/learning-parameters/mocks';
import { deepReplaceParameter } from './utils';

const getParam = (params: ConfigurableParameter[], key: string): ConfigurableParameter | undefined =>
    params.find((p) => p.key === key) as ConfigurableParameter | undefined;

const getGroup = (params: TrainingConfigurationParameter[], key: string): ConfigurableParameterGroup | undefined =>
    params.find((p) => isParameterGroup(p) && p.key === key) as ConfigurableParameterGroup | undefined;

describe('deepReplaceParameter', () => {
    describe('top-level parameter replacement', () => {
        it('replaces a top-level parameter by key', () => {
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'max_epochs', value: 50 });
            const result = deepReplaceParameter(learningParameters.parameters, updated);

            const replaced = getParam(result as ConfigurableParameter[], 'max_epochs');
            expect(replaced?.value).toBe(50);
        });

        it('does not affect other top-level parameters when replacing one', () => {
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'max_epochs', value: 99 });
            const result = deepReplaceParameter(learningParameters.parameters, updated);

            const batchSize = getParam(result as ConfigurableParameter[], 'batch_size');
            expect(batchSize?.value).toBe(4);
        });

        it('replaces a float top-level parameter', () => {
            const updated = getMockedConfigurationParameter({ value_type: 'float', key: 'learning_rate', value: 0.01 });
            const result = deepReplaceParameter(learningParameters.parameters, updated);

            const replaced = getParam(result as ConfigurableParameter[], 'learning_rate');
            expect(replaced?.value).toBe(0.01);
        });

        it('returns original array length unchanged', () => {
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'max_epochs', value: 1 });
            const result = deepReplaceParameter(learningParameters.parameters, updated);

            expect(result).toHaveLength(learningParameters.parameters.length);
        });

        it('does not mutate the original parameters array', () => {
            const original = learningParameters.parameters;
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'max_epochs', value: 1 });
            deepReplaceParameter(original, updated);

            const maxEpochs = getParam(original as ConfigurableParameter[], 'max_epochs');
            expect(maxEpochs?.value).toBe(200);
        });
    });

    describe('nested parameter replacement (no targetGroupKeys)', () => {
        it('replaces a parameter nested inside a group when no targetGroupKeys provided', () => {
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'patience', value: 99 });
            const result = deepReplaceParameter(learningParameters.parameters, updated);

            // early_stopping.patience should be replaced
            const earlyStoppingGroup = getGroup(result, 'early_stopping') as ConfigurableParameterGroup;
            const patience = getParam(earlyStoppingGroup.parameters as ConfigurableParameter[], 'patience');
            expect(patience?.value).toBe(99);
        });

        it('replaces all occurrences of a key across groups when no targetGroupKeys provided', () => {
            // Both early_stopping.patience and scheduler.patience exist
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'patience', value: 42 });
            const result = deepReplaceParameter(learningParameters.parameters, updated);

            const earlyStoppingGroup = getGroup(result, 'early_stopping') as ConfigurableParameterGroup;
            const earlyStoppingPatience = getParam(
                earlyStoppingGroup.parameters as ConfigurableParameter[],
                'patience'
            );
            expect(earlyStoppingPatience?.value).toBe(42);

            const schedulerGroup = getGroup(result, 'scheduler') as ConfigurableParameterGroup;
            const schedulerPatience = getParam(schedulerGroup.parameters as ConfigurableParameter[], 'patience');
            expect(schedulerPatience?.value).toBe(42);
        });

        it('replaces a bool parameter nested inside a group', () => {
            const updated = getMockedConfigurationParameter({ value_type: 'bool', key: 'enable', value: true });
            const result = deepReplaceParameter(learningParameters.parameters, updated);

            const gradientClipGroup = getGroup(result, 'gradient_clip') as ConfigurableParameterGroup;
            const enable = getParam(gradientClipGroup.parameters as ConfigurableParameter[], 'enable');
            expect(enable?.value).toBe(true);
        });
    });

    describe('targeted replacement using targetGroupKeys', () => {
        it('replaces only the parameter in the specific group when targetGroupKeys is provided', () => {
            // Replace patience only in early_stopping
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'patience', value: 77 });
            const result = deepReplaceParameter(learningParameters.parameters, updated, ['early_stopping']);

            const earlyStoppingGroup = getGroup(result, 'early_stopping') as ConfigurableParameterGroup;
            const earlyStoppingPatience = getParam(
                earlyStoppingGroup.parameters as ConfigurableParameter[],
                'patience'
            );
            expect(earlyStoppingPatience?.value).toBe(77);

            // scheduler.patience should remain unchanged
            const schedulerGroup = getGroup(result, 'scheduler') as ConfigurableParameterGroup;
            const schedulerPatience = getParam(schedulerGroup.parameters as ConfigurableParameter[], 'patience');
            expect(schedulerPatience?.value).toBe(5); // original value from mocks
        });

        it('replaces only the parameter in scheduler group when targetGroupKeys targets scheduler', () => {
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'patience', value: 55 });
            const result = deepReplaceParameter(learningParameters.parameters, updated, ['scheduler']);

            const schedulerGroup = getGroup(result, 'scheduler') as ConfigurableParameterGroup;
            const schedulerPatience = getParam(schedulerGroup.parameters as ConfigurableParameter[], 'patience');
            expect(schedulerPatience?.value).toBe(55);

            const earlyStoppingGroup = getGroup(result, 'early_stopping') as ConfigurableParameterGroup;
            const earlyStoppingPatience = getParam(
                earlyStoppingGroup.parameters as ConfigurableParameter[],
                'patience'
            );
            expect(earlyStoppingPatience?.value).toBe(10); // original value from mocks
        });

        it('replaces a deeply nested parameter using a multi-level targetGroupKeys path', () => {
            // scheduler.warmup.enable
            const updated = getMockedConfigurationParameter({ value_type: 'bool', key: 'enable', value: true });
            const result = deepReplaceParameter(learningParameters.parameters, updated, ['scheduler', 'warmup']);

            const schedulerGroup = getGroup(result, 'scheduler') as ConfigurableParameterGroup;
            const warmupGroup = getGroup(schedulerGroup.parameters, 'warmup') as ConfigurableParameterGroup;
            const enable = getParam(warmupGroup.parameters as ConfigurableParameter[], 'enable');
            expect(enable?.value).toBe(true);

            // gradient_accumulation.enable should remain unchanged
            const gradAccGroup = getGroup(result, 'gradient_accumulation') as ConfigurableParameterGroup;
            const gradAccEnable = getParam(gradAccGroup.parameters as ConfigurableParameter[], 'enable');
            expect(gradAccEnable?.value).toBe(false);

            // gradient_clip.enable should remain unchanged
            const gradClipGroup = getGroup(result, 'gradient_clip') as ConfigurableParameterGroup;
            const gradClipEnable = getParam(gradClipGroup.parameters as ConfigurableParameter[], 'enable');
            expect(gradClipEnable?.value).toBe(false);
        });

        it('does not replace parameter when targetGroupKeys path does not match actual nesting', () => {
            // Try to replace max_epochs using a wrong group key
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'max_epochs', value: 1 });
            const result = deepReplaceParameter(learningParameters.parameters, updated, ['nonexistent_group']);

            const maxEpochs = getParam(result as ConfigurableParameter[], 'max_epochs');
            // max_epochs is top-level so currentGroupKeys would be [] which does not match ['nonexistent_group']
            expect(maxEpochs?.value).toBe(200);
        });

        it('does not replace parameter when targetGroupKeys length mismatches current depth', () => {
            // patience is at depth 1 inside early_stopping, passing two-level path should not match
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'patience', value: 1 });
            const result = deepReplaceParameter(learningParameters.parameters, updated, ['training', 'early_stopping']);

            const earlyStoppingGroup = getGroup(result, 'early_stopping') as ConfigurableParameterGroup;
            const patience = getParam(earlyStoppingGroup.parameters as ConfigurableParameter[], 'patience');
            expect(patience?.value).toBe(10); // unchanged
        });
    });

    describe('edge cases', () => {
        it('returns unchanged array when key does not exist in parameters', () => {
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'nonexistent_key', value: 1 });
            const result = deepReplaceParameter(learningParameters.parameters, updated);

            // All original values should remain
            const maxEpochs = getParam(result as ConfigurableParameter[], 'max_epochs');
            expect(maxEpochs?.value).toBe(200);
        });

        it('handles an empty parameters array', () => {
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'max_epochs', value: 1 });
            const result = deepReplaceParameter([], updated);

            expect(result).toEqual([]);
        });

        it('handles a flat list of parameters with no groups', () => {
            const params = [
                getMockedConfigurationParameter({ value_type: 'int', key: 'alpha', value: 1 }),
                getMockedConfigurationParameter({ value_type: 'int', key: 'beta', value: 2 }),
            ];
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'alpha', value: 99 });
            const result = deepReplaceParameter(params, updated);

            expect((result[0] as ConfigurableParameter).value).toBe(99);
            expect((result[1] as ConfigurableParameter).value).toBe(2);
        });

        it('handles a list with only groups (no top-level leaf parameters)', () => {
            const params = [
                getMockedConfigurationParameterGroup({
                    key: 'group_a',
                    parameters: [getMockedConfigurationParameter({ value_type: 'int', key: 'x', value: 10 })],
                }),
            ];
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'x', value: 42 });
            const result = deepReplaceParameter(params, updated);

            const groupA = getGroup(result, 'group_a') as ConfigurableParameterGroup;
            expect((groupA.parameters[0] as ConfigurableParameter).value).toBe(42);
        });

        it('preserves all other fields of the group when replacing a nested parameter', () => {
            const updated = getMockedConfigurationParameter({ value_type: 'bool', key: 'enable', value: true });
            const result = deepReplaceParameter(learningParameters.parameters, updated, ['gradient_accumulation']);

            const gradAccGroup = getGroup(result, 'gradient_accumulation') as ConfigurableParameterGroup;
            expect(gradAccGroup.name).toBe('Gradient accumulation');
            expect(gradAccGroup.description).toContain('Gradient accumulation allows');
            expect(gradAccGroup.type).toBe('parameter_group');
        });

        it('does not replace the parameter when targetGroupKeys is an empty array', () => {
            // Empty targetGroupKeys means targetGroupKeys.length === 0, so the
            // `if (targetGroupKeys !== undefined && targetGroupKeys.length > 0)` branch is skipped
            // and matching falls through to the key-only comparison
            const updated = getMockedConfigurationParameter({ value_type: 'int', key: 'max_epochs', value: 1 });
            const result = deepReplaceParameter(learningParameters.parameters, updated, []);

            const maxEpochs = getParam(result as ConfigurableParameter[], 'max_epochs');
            expect(maxEpochs?.value).toBe(1);
        });

        it('replaces parameter at top level when targetGroupKeys is undefined', () => {
            const updated = getMockedConfigurationParameter({ value_type: 'float', key: 'weight_decay', value: 0.001 });
            const result = deepReplaceParameter(learningParameters.parameters, updated, undefined);

            const replaced = getParam(result as ConfigurableParameter[], 'weight_decay');
            expect(replaced?.value).toBe(0.001);
        });
    });
});
