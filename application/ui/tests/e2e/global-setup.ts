// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

async function globalSetup() {
    // Copy video file from test assets to backend media directory
    const sourceVideo = path.join(dirname, '../assets/test_video.mp4');
    const targetVideo = path.join(dirname, '../../../backend/data/media/video.mp4');

    const targetDir = path.dirname(targetVideo);
    if (!fs.existsSync(targetDir)) {
        fs.mkdirSync(targetDir, { recursive: true });
    }

    fs.copyFileSync(sourceVideo, targetVideo);
    console.info('✓ Test video file copied to backend media directory');
}

export default globalSetup;
