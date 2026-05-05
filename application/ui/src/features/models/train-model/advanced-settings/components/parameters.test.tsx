// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import { render } from 'test-utils/render';

import { StringEnumConfigurableParameter } from '../../../../../constants/shared-types';
import { Parameter } from './parameters.component';

const createStringEnumParameter = (
    overrides: Partial<StringEnumConfigurableParameter> = {}
): StringEnumConfigurableParameter => ({
    type: 'parameter',
    key: 'test_param',
    name: 'Test parameter',
    description: 'A test parameter',
    value_type: 'str',
    value: 'value_a',
    default_value: 'value_a',
    allowed_values: ['value_a', 'value_b'],
    ...overrides,
});

describe('Parameter', () => {
    describe('StringEnum display names', () => {
        it('shows display names when allowed_values_display_names is provided', () => {
            const parameter = createStringEnumParameter({
                allowed_values_display_names: { value_a: 'Display A', value_b: 'Display B' },
            });

            render(
                <Parameter
                    header={parameter.name}
                    description={parameter.description}
                    parameter={parameter}
                    onChange={vi.fn()}
                    isReadOnly={false}
                />
            );

            expect(screen.getByText('Display A')).toBeVisible();
        });

        it('falls back to the raw value when allowed_values_display_names is not provided', () => {
            const parameter = createStringEnumParameter({
                allowed_values: ['value_a'],
                allowed_values_display_names: undefined,
            });

            render(
                <Parameter
                    header={parameter.name}
                    description={parameter.description}
                    parameter={parameter}
                    onChange={vi.fn()}
                    isReadOnly={false}
                />
            );

            expect(screen.queryByText('Display A')).not.toBeInTheDocument();
            expect(screen.getAllByText('value_a').length).toBeGreaterThanOrEqual(1);
        });

        it('falls back to the raw value when the key is missing from allowed_values_display_names', () => {
            const parameter = createStringEnumParameter({
                allowed_values: ['value_a'],
                allowed_values_display_names: { value_b: 'Display B' },
            });

            render(
                <Parameter
                    header={parameter.name}
                    description={parameter.description}
                    parameter={parameter}
                    onChange={vi.fn()}
                    isReadOnly={false}
                />
            );

            expect(screen.queryByText('Display A')).not.toBeInTheDocument();
            expect(screen.getAllByText('value_a').length).toBeGreaterThanOrEqual(1);
        });
    });
});
