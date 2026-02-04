// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ColorEditor, ColorSwatch, ColorSwatchPicker, Flex, ColorPicker as SpectrumColorPicker } from '@geti/ui';

import { DISTINCT_COLORS } from '../../features/annotator/label-utils';

interface LabelColorPickerProps {
    color: string;
    onChange: (color: string) => void;
}

export const LabelColorPicker = ({ color, onChange }: LabelColorPickerProps) => {
    return (
        <SpectrumColorPicker value={color} onChange={(c) => onChange(c.toString('hex'))} rounding={'none'}>
            <Flex direction='column' gap='size-300'>
                <ColorEditor />
                <ColorSwatchPicker width={'size-3600'}>
                    {DISTINCT_COLORS.map((presetColor) => (
                        <ColorSwatch color={presetColor} key={presetColor} />
                    ))}
                </ColorSwatchPicker>
            </Flex>
        </SpectrumColorPicker>
    );
};
