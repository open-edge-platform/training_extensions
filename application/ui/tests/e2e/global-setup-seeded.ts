// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import https from 'https';
import path from 'path';
import { fileURLToPath } from 'url';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

// S3 bucket configuration
const S3_BUCKET_URL = process.env.E2E_ASSETS_S3_URL || 'https://storage.geti.intel.com/test-data/geti-tune';

// Model files needed for seeded tests
const MODEL_FILES = ['977eeb18-eaac-449d-bc80-e340fbe052ad.bin', '977eeb18-eaac-449d-bc80-e340fbe052ad.xml'];

/**
 * Download a file from URL to local path
 */
async function downloadFile(url: string, destPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(destPath);

        https
            .get(url, (response) => {
                if (response.statusCode === 200) {
                    response.pipe(file);
                    file.on('finish', () => {
                        file.close();
                        resolve();
                    });
                } else {
                    file.close();
                    fs.unlinkSync(destPath);
                    reject(new Error(`Failed to download ${url}: ${response.statusCode}`));
                }
            })
            .on('error', (err) => {
                file.close();
                fs.unlinkSync(destPath);
                reject(err);
            });
    });
}

/**
 * Global setup for seeded E2E tests
 * Downloads/copies video and model files
 */
async function globalSetupSeeded() {
    // Setup directories
    const targetVideo = path.join(dirname, '../../../backend/data/media/video.mp4');
    const modelsDir = path.join(dirname, '../../../backend/data/models');

    const mediaDir = path.dirname(targetVideo);
    if (!fs.existsSync(mediaDir)) {
        fs.mkdirSync(mediaDir, { recursive: true });
    }
    if (!fs.existsSync(modelsDir)) {
        fs.mkdirSync(modelsDir, { recursive: true });
    }

    if (!!process.env.CI) {
        console.info('Downloading test assets from S3...');

        // Download video
        try {
            const videoUrl = `${S3_BUCKET_URL}/test_video.mp4`;
            await downloadFile(videoUrl, targetVideo);
            console.info('✓ Test video downloaded from S3');
        } catch (error) {
            console.error('✗ Failed to download video from S3:', error);
            throw error;
        }

        // Download model files
        for (const modelFile of MODEL_FILES) {
            try {
                const modelUrl = `${S3_BUCKET_URL}/models/${modelFile}`;
                const targetPath = path.join(modelsDir, modelFile);
                await downloadFile(modelUrl, targetPath);
                console.info(`✓ Downloaded model: ${modelFile}`);
            } catch (error) {
                console.error(`✗ Failed to download model ${modelFile}:`, error);
                throw error;
            }
        }
    } else {
        // Use local files
        console.info('Using local test assets...');

        // Copy video
        const sourceVideo = path.join(dirname, '../assets/test_video.mp4');
        if (!fs.existsSync(sourceVideo)) {
            throw new Error(
                `Test video not found at ${sourceVideo}. ` + 'Either place the file locally or set USE_S3_ASSETS=true'
            );
        }
        fs.copyFileSync(sourceVideo, targetVideo);
        console.info('✓ Test video copied from local assets');

        // Copy model files
        const modelsSourceDir = path.join(dirname, '../assets/models');
        if (fs.existsSync(modelsSourceDir)) {
            for (const modelFile of MODEL_FILES) {
                const sourcePath = path.join(modelsSourceDir, modelFile);
                const targetPath = path.join(modelsDir, modelFile);

                if (fs.existsSync(sourcePath)) {
                    fs.copyFileSync(sourcePath, targetPath);
                    console.info(`✓ Copied model: ${modelFile}`);
                } else {
                    throw new Error(`Model file not found: ${sourcePath}.`);
                }
            }
        } else {
            throw new Error(`Models directory not found at ${modelsSourceDir}.`);
        }
    }
}

export default globalSetupSeeded;
