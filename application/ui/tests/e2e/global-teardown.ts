// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

const VIDEO_FILE = 'video.mp4';
const MODEL_FILES = ['ssd-card-detection.bin', 'ssd-card-detection.xml'];

function resolveBackendPath(subpath: string): string {
    return path.join(dirname, '../../../backend/data', subpath);
}

async function globalTeardown() {
    console.info('=== Global Teardown: Cleaning up test assets ===');

    const mediaDir = resolveBackendPath('media');
    const modelsDir = resolveBackendPath('models');
    const targetVideo = path.join(mediaDir, VIDEO_FILE);

    // Remove video file
    if (fs.existsSync(targetVideo)) {
        try {
            fs.unlinkSync(targetVideo);
            console.info(`✓ Removed test video: ${VIDEO_FILE}`);
        } catch (error) {
            console.warn(`⚠ Failed to remove test video: ${error}`);
        }
    }

    // Remove model files
    for (const modelFile of MODEL_FILES) {
        const modelPath = path.join(modelsDir, modelFile);
        if (fs.existsSync(modelPath)) {
            try {
                fs.unlinkSync(modelPath);
                console.info(`✓ Removed model file: ${modelFile}`);
            } catch (error) {
                console.warn(`⚠ Failed to remove model file ${modelFile}: ${error}`);
            }
        }
    }

    console.info('=== Global Teardown: Complete ===');
}

export default globalTeardown;
