// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from 'fs';
import http from 'http';
import https from 'https';
import path from 'path';
import { fileURLToPath } from 'url';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

// S3 bucket configuration
const S3_BUCKET_URL = process.env.E2E_ASSETS_S3_URL;
const VIDEO_FILE = 'sample-video-small.mp4';
const MODEL_FILES = ['ssd-card-detection.bin', 'ssd-card-detection.xml'];

function resolveBackendPath(subpath: string): string {
    return path.join(dirname, '../../../backend/data', subpath);
}

async function downloadFile(url: string, destPath: string): Promise<void> {
    console.info(`Downloading ${url}...`);

    return new Promise((resolve, reject) => {
        const parsedUrl = new URL(url);
        const client = parsedUrl.protocol === 'https:' ? https : http;
        const fileStream = fs.createWriteStream(destPath);

        const request = client.get(url, (response) => {
            // Handle redirects
            if (response.statusCode === 301 || response.statusCode === 302) {
                const redirectUrl = response.headers.location;
                if (redirectUrl) {
                    fileStream.close();
                    fs.unlinkSync(destPath);
                    downloadFile(redirectUrl, destPath).then(resolve).catch(reject);
                    return;
                }
            }

            if (response.statusCode === 200) {
                response.pipe(fileStream);
                fileStream.on('finish', () => {
                    fileStream.close();
                    console.info(`✓ Downloaded ${path.basename(destPath)}`);
                    resolve();
                });
            } else {
                fileStream.close();
                if (fs.existsSync(destPath)) {
                    fs.unlinkSync(destPath);
                }
                reject(new Error(`Failed to download ${url}: HTTP ${response.statusCode}`));
            }
        });

        request.on('error', (error) => {
            fileStream.close();
            if (fs.existsSync(destPath)) {
                fs.unlinkSync(destPath);
            }
            reject(error);
        });

        fileStream.on('error', (error) => {
            request.destroy();
            if (fs.existsSync(destPath)) {
                fs.unlinkSync(destPath);
            }
            reject(error);
        });
    });
}

async function globalSetup() {
    console.info('=== Global Setup: Preparing test assets ===');

    const mediaDir = resolveBackendPath('media');
    const modelsDir = resolveBackendPath('models');
    const targetVideo = path.join(mediaDir, 'sample-video-small.mp4');

    // Create directories
    if (!fs.existsSync(mediaDir)) {
        fs.mkdirSync(mediaDir, { recursive: true });
    }
    if (!fs.existsSync(modelsDir)) {
        fs.mkdirSync(modelsDir, { recursive: true });
    }

    // Check if assets already exist
    const videoExists = fs.existsSync(targetVideo);
    const modelsExist = MODEL_FILES.every((file) => fs.existsSync(path.join(modelsDir, file)));

    if (videoExists && modelsExist) {
        console.info('✓ Test assets already exist, skipping download');
        return;
    }

    // Download from S3 or use local files
    if (S3_BUCKET_URL) {
        console.info(`Downloading test assets from ${S3_BUCKET_URL}...`);

        // Download video
        if (!videoExists) {
            const videoUrl = `${S3_BUCKET_URL}/media/${VIDEO_FILE}`;
            await downloadFile(videoUrl, targetVideo);
        }

        // Download model files
        for (const modelFile of MODEL_FILES) {
            const targetPath = path.join(modelsDir, modelFile);
            if (!fs.existsSync(targetPath)) {
                const modelUrl = `${S3_BUCKET_URL}/models/${modelFile}`;
                await downloadFile(modelUrl, targetPath);
            }
        }

        console.info('✓ All test assets downloaded from S3');
    } else {
        // Use local files
        console.info('Using local test assets (E2E_ASSETS_S3_URL not set)...');

        // Copy video
        if (!videoExists) {
            const sourceVideo = path.join(dirname, '../assets', VIDEO_FILE);
            if (!fs.existsSync(sourceVideo)) {
                throw new Error(
                    `Test video not found at ${sourceVideo}. ` +
                        `Either place ${VIDEO_FILE} in tests/assets/ or set E2E_ASSETS_S3_URL environment variable.`
                );
            }
            fs.copyFileSync(sourceVideo, targetVideo);
            console.info('✓ Test video copied from local assets');
        }

        // Copy model files
        const modelsSourceDir = path.join(dirname, '../assets/models');
        if (!fs.existsSync(modelsSourceDir)) {
            throw new Error(
                `Models directory not found at ${modelsSourceDir}. ` +
                    `Either create tests/assets/models/ with required files ` +
                    `or set E2E_ASSETS_S3_URL environment variable.`
            );
        }

        for (const modelFile of MODEL_FILES) {
            const targetPath = path.join(modelsDir, modelFile);
            if (!fs.existsSync(targetPath)) {
                const sourcePath = path.join(modelsSourceDir, modelFile);
                if (!fs.existsSync(sourcePath)) {
                    throw new Error(`Model file not found: ${sourcePath}`);
                }
                fs.copyFileSync(sourcePath, targetPath);
                console.info(`✓ Copied model: ${modelFile}`);
            }
        }

        console.info('✓ All test assets copied from local files');
    }

    console.info('=== Global Setup: Complete ===');
}

export default globalSetup;
