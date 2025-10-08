// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import https from 'https';
import path from 'path';
import { fileURLToPath } from 'url';

import { expect, test } from '@playwright/test';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

const BACKEND_DIR = path.resolve(dirname, '../../../../backend');
const DATA_DIR = path.join(BACKEND_DIR, 'data');
const MEDIA_DIR = path.join(DATA_DIR, 'media');

// Model directory structure: data/projects/{project_id}/models/{model_id}/
const PROJECT_ID = '9d6af8e8-6017-4ebe-9126-33aae739c5fa';
const MODEL_ID = '977eeb18-eaac-449d-bc80-e340fbe052ad';
const MODELS_DIR = path.join(DATA_DIR, 'projects', PROJECT_ID, 'models', MODEL_ID);

const E2E_ASSETS_S3_URL = process.env.E2E_ASSETS_S3_URL;

async function downloadFile(url: string, destPath: string): Promise<void> {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(destPath);
        https
            .get(url, (response) => {
                if (response.statusCode === 302 || response.statusCode === 301) {
                    const redirectUrl = response.headers.location;

                    if (!redirectUrl) {
                        reject(new Error('Redirect location not found'));
                        return;
                    }

                    https.get(redirectUrl, (redirectResponse) => {
                        redirectResponse.pipe(file);
                        file.on('finish', () => {
                            file.close();
                            resolve();
                        });
                    });
                } else if (response.statusCode === 200) {
                    response.pipe(file);
                    file.on('finish', () => {
                        file.close();
                        resolve();
                    });
                } else {
                    reject(new Error(`Failed to download: ${response.statusCode}`));
                }
            })
            .on('error', (err) => {
                fs.unlink(destPath, () => reject(err));
            });
    });
}

test.describe('E2E Setup', () => {
    test('Download test assets', async () => {
        console.info('=============================================');
        console.info('=== E2E SETUP: Downloading test assets ===');
        console.info('=============================================');
        console.info('BACKEND_DIR:', BACKEND_DIR);
        console.info('E2E_ASSETS_S3_URL:', E2E_ASSETS_S3_URL || 'not set');
        console.info('=============================================');

        // Create directories
        fs.mkdirSync(MEDIA_DIR, { recursive: true });
        fs.mkdirSync(MODELS_DIR, { recursive: true });

        if (!E2E_ASSETS_S3_URL) {
            console.info('⚠️  E2E_ASSETS_S3_URL not set, checking for local files');

            // Verify local files exist
            const videoPath = path.join(MEDIA_DIR, 'video.mp4');
            const modelXmlPath = path.join(MODELS_DIR, 'model.xml');
            const modelBinPath = path.join(MODELS_DIR, 'model.bin');

            expect(fs.existsSync(videoPath), `Video file not found: ${videoPath}`).toBeTruthy();
            expect(fs.existsSync(modelXmlPath), `Model XML not found: ${modelXmlPath}`).toBeTruthy();
            expect(fs.existsSync(modelBinPath), `Model BIN not found: ${modelBinPath}`).toBeTruthy();

            console.info('✓ All required assets found locally');
            return;
        }

        console.info('Downloading test assets from S3:', E2E_ASSETS_S3_URL);

        // Download video
        const videoPath = path.join(MEDIA_DIR, 'video.mp4');
        if (!fs.existsSync(videoPath)) {
            console.info('Downloading video.mp4...');
            await downloadFile(`${E2E_ASSETS_S3_URL}/media/sample-video-small.mp4`, videoPath);
            console.info('✓ Downloaded video.mp4');
        } else {
            console.info('✓ video.mp4 already exists');
        }

        // Download model.xml
        const modelXmlPath = path.join(MODELS_DIR, 'model.xml');
        if (!fs.existsSync(modelXmlPath)) {
            console.info('Downloading model.xml...');
            await downloadFile(`${E2E_ASSETS_S3_URL}/models/ssd-card-detection.xml`, modelXmlPath);
            console.info('✓ Downloaded model.xml');
        } else {
            console.info('✓ model.xml already exists');
        }

        // Download model.bin
        const modelBinPath = path.join(MODELS_DIR, 'model.bin');
        if (!fs.existsSync(modelBinPath)) {
            console.info('Downloading model.bin...');
            await downloadFile(`${E2E_ASSETS_S3_URL}/models/ssd-card-detection.bin`, modelBinPath);
            console.info('✓ Downloaded model.bin');
        } else {
            console.info('✓ model.bin already exists');
        }

        // Verify all files exist
        expect(fs.existsSync(videoPath), 'Video file was not downloaded').toBeTruthy();
        expect(fs.existsSync(modelXmlPath), 'Model XML was not downloaded').toBeTruthy();
        expect(fs.existsSync(modelBinPath), 'Model BIN was not downloaded').toBeTruthy();

        console.info('✓ All test assets ready');
        console.info('=============================================');
    });
});
