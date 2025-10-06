// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

async function globalTeardown() {
    const targetVideo = path.join(dirname, '../../../backend/data/media/test_video.mp4');

    if (fs.existsSync(targetVideo)) {
        fs.unlinkSync(targetVideo);
        console.info('✓ Test video file cleaned up');
    }
}

export default globalTeardown;
