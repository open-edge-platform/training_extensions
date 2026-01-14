// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Divider, Flex, Switch, Text, View } from '@geti/ui';

import { AnnotationSetting } from './annotation-setting.component';
import { CanvasSettingsState } from './canvas-settings-provider.component';
import { ImageSetting } from './image-setting.component';

interface SettingsListProps {
    canvasSettings: CanvasSettingsState;
    onCanvasSettingsChange: (canvasSettings: CanvasSettingsState) => void;
}

export const SettingsList = ({ canvasSettings, onCanvasSettingsChange }: SettingsListProps) => {
    const updateCanvasSettings = <T extends keyof CanvasSettingsState>(
        key: T,
        value: CanvasSettingsState[T]['value']
    ) => {
        const newCanvasSettings = structuredClone(canvasSettings);
        newCanvasSettings[key].value = value;
        onCanvasSettingsChange(newCanvasSettings);
    };

    return (
        <View paddingEnd={'size-50'}>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Text>Hide labels</Text>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Switch
                        aria-label={'Hide labels'}
                        isEmphasized
                        isSelected={canvasSettings.hideLabels.value}
                        onChange={(isSelected) => {
                            updateCanvasSettings('hideLabels', isSelected);
                        }}
                    />
                </Flex>
            </Flex>

            <Divider size={'S'} marginY={'size-250'} />

            <AnnotationSetting
                headerText={'Annotation fill opacity'}
                formatOptions={{ style: 'percent' }}
                defaultValue={canvasSettings.annotationFillOpacity.defaultValue}
                value={canvasSettings.annotationFillOpacity.value}
                handleValueChange={(value) => {
                    updateCanvasSettings('annotationFillOpacity', value);
                }}
            />
            <AnnotationSetting
                headerText={'Annotation border opacity'}
                formatOptions={{ style: 'percent' }}
                defaultValue={canvasSettings.annotationBorderOpacity.defaultValue}
                value={canvasSettings.annotationBorderOpacity.value}
                handleValueChange={(value) => {
                    updateCanvasSettings('annotationBorderOpacity', value);
                }}
            />
            <Divider size={'S'} marginY={'size-250'} />
            <ImageSetting
                headerText={'Image brightness'}
                formatOptions={{ signDisplay: 'exceptZero' }}
                defaultValue={canvasSettings.imageBrightness.defaultValue}
                value={canvasSettings.imageBrightness.value}
                handleValueChange={(value) => {
                    updateCanvasSettings('imageBrightness', value);
                }}
            />
            <ImageSetting
                headerText={'Image contrast'}
                formatOptions={{ signDisplay: 'exceptZero' }}
                defaultValue={canvasSettings.imageContrast.defaultValue}
                value={canvasSettings.imageContrast.value}
                handleValueChange={(value) => {
                    updateCanvasSettings('imageContrast', value);
                }}
            />
            <ImageSetting
                headerText={'Image saturation'}
                formatOptions={{ signDisplay: 'exceptZero' }}
                defaultValue={canvasSettings.imageSaturation.defaultValue}
                value={canvasSettings.imageSaturation.value}
                handleValueChange={(value) => {
                    updateCanvasSettings('imageSaturation', value);
                }}
            />
            <Divider size={'S'} marginY={'size-250'} />

            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Text>Pixel view</Text>
                <Flex alignItems={'center'} gap={'size-100'}>
                    <Switch
                        aria-label={'Pixel view'}
                        isEmphasized
                        isSelected={canvasSettings.pixelView.value}
                        onChange={(isSelected) => {
                            updateCanvasSettings('pixelView', isSelected);
                        }}
                    />
                </Flex>
            </Flex>
        </View>
    );
};
