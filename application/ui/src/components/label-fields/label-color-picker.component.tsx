// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ColorEditor, ColorSwatch, ColorSwatchPicker, Flex, ColorPicker as SpectrumColorPicker } from '@geti-ui/ui';

import { DISTINCT_COLORS } from '../../features/annotator/label-utils';

type LabelColorPickerProps = {
    color: string;
    onChange: (color: string) => void;
};

export const LabelColorPicker = ({ color, onChange }: LabelColorPickerProps) => {
    return (
        <SpectrumColorPicker
            value={color}
            onChange={(newColor) => onChange(newColor.toString('hex'))}
            rounding={'none'}
        >
            <Flex direction='column' gap='size-300'>
                <ColorEditor hideAlphaChannel />
                <ColorSwatchPicker width={'size-3000'}>
                    {DISTINCT_COLORS.map((presetColor) => (
                        <ColorSwatch color={presetColor} key={presetColor} />
                    ))}
                </ColorSwatchPicker>
            </Flex>
        </SpectrumColorPicker>
    );
};
