// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Checkbox, CheckboxGroup } from '@geti/ui';

import { OutputFormat } from './utils';

import classes from './output-formats.module.scss';

export const OutputFormats = () => {
    return (
        <CheckboxGroup label='Output Formats' UNSAFE_className={classes.checkboxGroup}>
            <Checkbox name='output_formats' value={OutputFormat.PREDICTIONS}>
                Predictions
            </Checkbox>
            <Checkbox name='output_formats' value={OutputFormat.IMAGE_ORIGINAL}>
                Image Original
            </Checkbox>
            <Checkbox name='output_formats' value={OutputFormat.IMAGE_WITH_PREDICTIONS}>
                Image with Predictions
            </Checkbox>
        </CheckboxGroup>
    );
};
