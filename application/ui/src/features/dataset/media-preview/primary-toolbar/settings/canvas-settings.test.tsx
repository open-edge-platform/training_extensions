// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen, within } from '@testing-library/react';

import { CanvasSettingsProvider, DEFAULT_CANVAS_SETTINGS } from './canvas-settings-provider.component';
import { CanvasSettings } from './canvas-settings.component';

const getContainer = (setting: string) => within(screen.getByLabelText(setting));

const increaseSetting = (setting: string, count: number) => {
    const container = getContainer(setting);

    for (let i = 0; i < count; i++) {
        fireEvent.keyDown(container.getByRole('slider'), { key: 'Right' });
    }
};

const decreaseSetting = (setting: string, count: number) => {
    const container = getContainer(setting);

    for (let i = 0; i < count; i++) {
        fireEvent.keyDown(container.getByRole('slider'), { key: 'Left' });
    }
};

const expectSettingValueToBe = (setting: string, value: string) => {
    const container = getContainer(setting);

    expect(container.getByText(value)).toBeInTheDocument();
};

const resetSetting = (setting: string) => {
    fireEvent.click(screen.getByRole('button', { name: new RegExp(`Reset ${setting}`, 'i') }));
};

describe('CanvasSettings', () => {
    const renderCanvasAdjustments = () => {
        render(
            <CanvasSettingsProvider>
                <CanvasSettings />
            </CanvasSettingsProvider>
        );
    };

    it('Updates annotation fill opacity and resets to default', () => {
        renderCanvasAdjustments();

        const setting = 'Annotation fill opacity';

        const increaseBy = 10;

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.annotationFillOpacity.value * 100}%`);

        increaseSetting(setting, increaseBy);

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.annotationFillOpacity.value * 100 + increaseBy}%`);

        resetSetting(setting);

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.annotationFillOpacity.defaultValue * 100}%`);
    });

    it('Updates annotation border opacity and resets to default', () => {
        renderCanvasAdjustments();

        const setting = 'Annotation border opacity';

        const decreaseBy = 5;

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.annotationBorderOpacity.value * 100}%`);

        decreaseSetting(setting, decreaseBy);

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.annotationBorderOpacity.value * 100 - decreaseBy}%`);

        resetSetting(setting);

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.annotationBorderOpacity.defaultValue * 100}%`);
    });

    it('Updates image brightness and resets to default', () => {
        renderCanvasAdjustments();

        const setting = 'Image brightness';

        const increaseBy = 5;

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.imageBrightness.value}`);

        increaseSetting(setting, increaseBy);

        expectSettingValueToBe(setting, `+${DEFAULT_CANVAS_SETTINGS.imageBrightness.value + increaseBy}`);

        resetSetting(setting);

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.imageBrightness.defaultValue}`);
    });

    it('Updates image saturation and resets to default', () => {
        renderCanvasAdjustments();

        const setting = 'Image saturation';

        const increaseBy = 5;

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.imageSaturation.value}`);

        increaseSetting(setting, increaseBy);

        expectSettingValueToBe(setting, `+${DEFAULT_CANVAS_SETTINGS.imageSaturation.value + increaseBy}`);

        resetSetting(setting);

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.imageSaturation.defaultValue}`);
    });

    it('Updates image contrast and resets to default', () => {
        renderCanvasAdjustments();

        const setting = 'Image contrast';

        const increaseBy = 5;

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.imageContrast.value}`);

        increaseSetting(setting, increaseBy);

        expectSettingValueToBe(setting, `+${DEFAULT_CANVAS_SETTINGS.imageContrast.value + increaseBy}`);

        resetSetting(setting);

        expectSettingValueToBe(setting, `${DEFAULT_CANVAS_SETTINGS.imageContrast.defaultValue}`);
    });

    it('Updates pixel view and resets to default', () => {
        renderCanvasAdjustments();

        const pixelView = screen.getByRole('switch', { name: 'Pixel view' });

        expect(pixelView).not.toBeChecked();

        fireEvent.click(pixelView);

        expect(pixelView).toBeChecked();
    });
});
