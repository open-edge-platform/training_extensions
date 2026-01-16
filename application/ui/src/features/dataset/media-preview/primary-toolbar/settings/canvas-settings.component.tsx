// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCanvasSettings } from './canvas-settings-provider.component';
import { SettingsList } from './settings-list.component';

export const CanvasSettings = () => {
    const { canvasSettings, setCanvasSettings } = useCanvasSettings();

    return <SettingsList canvasSettings={canvasSettings} onCanvasSettingsChange={setCanvasSettings} />;
};
