import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { chromium } from 'playwright';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const baseURL = process.env.BASE_URL ?? 'http://127.0.0.1:3000';
const videoDir = path.join(dirname, 'videos');
const finalVideoPath = path.join(videoDir, 'video-stream-initial-load-resize.webm');
const measurementsPath = path.join(dirname, 'video-stream-initial-load-resize.measurements.json');
const screenshotPath = path.join(dirname, 'video-stream-final.png');

const project = {
    id: 'id-1',
    name: 'Video stream resize project',
    task: {
        task_type: 'detection',
        exclusive_labels: false,
        labels: [{ id: 'label-1', name: 'Object', color: '#00AEEF', hotkey: 'O' }],
    },
    active_pipeline: true,
    created_at: '2026-06-24T00:00:00Z',
};

const pipeline = {
    project_id: project.id,
    status: 'running',
    source: {
        id: 'source-id',
        name: 'Synthetic stream',
        source_type: 'video_file',
        video_path: 'synthetic-stream.mp4',
        loop: true,
    },
    model: {
        id: 'model-id',
        name: 'Synthetic detection model',
        architecture: 'Object_Detection_TestModel',
        training_info: {
            status: 'successful',
            label_schema_revision: {},
        },
        files_deleted: false,
        variants: [],
    },
    sink: {
        id: 'sink-id',
        name: 'Local folder',
        sink_type: 'folder',
        folder_path: 'data/output',
        output_formats: ['image_original', 'image_with_predictions', 'predictions'],
        rate_limit: 0.2,
    },
    model_variant: null,
    device: 'cpu',
    data_collection: {
        max_dataset_size: 500,
        policies: [
            { type: 'fixed_rate', enabled: true, rate: 12 },
            { type: 'confidence_threshold', enabled: false, confidence_threshold: 0.5, min_sampling_interval: 2.5 },
        ],
    },
};

const fulfillJson = (route, data, status = 200) => {
    return route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(data),
    });
};

const apiResponseFor = (pathname, method) => {
    if (pathname === '/api/system/info') {
        return { data: { license_accepted: true, platform: 'windows' } };
    }
    if (pathname === '/api/system/metrics/memory') {
        return { data: {} };
    }
    if (pathname === '/api/projects') {
        return { data: [project] };
    }
    if (pathname === `/api/projects/${project.id}`) {
        return { data: project };
    }
    if (pathname === `/api/projects/${project.id}/pipeline`) {
        return { data: pipeline };
    }
    if (pathname === `/api/projects/${project.id}/pipeline/metrics`) {
        return {
            data: {
                inference: {
                    latency: { avg_ms: 11.4 },
                    throughput: { avg_requests_per_second: 27.2 },
                },
            },
        };
    }
    if (pathname === `/api/projects/${project.id}/models`) {
        return { data: [] };
    }
    if (pathname === '/api/system/devices/inference') {
        return { data: [{ type: 'cpu', name: 'CPU' }] };
    }
    if (pathname === '/api/sources') {
        return { data: [pipeline.source] };
    }
    if (pathname === '/api/sinks') {
        return { data: [pipeline.sink] };
    }
    if (pathname === '/api/webrtc/config') {
        return { data: { iceServers: [] } };
    }
    if (pathname === '/api/webrtc/offer' && method === 'POST') {
        return { data: { type: 'answer', sdp: 'v=0\r\n' } };
    }
    if (pathname === '/api/webrtc/input_hook' && method === 'POST') {
        return { data: {} };
    }
    if (pathname === `/api/projects/${project.id}/pipeline:enable` && method === 'POST') {
        return { data: null, status: 204 };
    }
    if (pathname === `/api/projects/${project.id}/pipeline:disable` && method === 'POST') {
        return { data: null, status: 204 };
    }

    return { data: {}, status: 200 };
};

const syntheticWebRTCScript = () => {
    const ensureSyntheticStream = () => {
        if (window.__getiSyntheticStreamState) {
            return window.__getiSyntheticStreamState.stream;
        }

        const canvas = document.createElement('canvas');
        const state = {
            width: 640,
            height: 480,
            label: 'initial stream 640 x 480',
            frame: 0,
            canvas,
            stream: undefined,
        };

        const draw = () => {
            const context = canvas.getContext('2d');
            const width = state.width;
            const height = state.height;
            state.frame += 1;

            canvas.width = width;
            canvas.height = height;

            const gradient = context.createLinearGradient(0, 0, width, height);
            gradient.addColorStop(0, '#0f172a');
            gradient.addColorStop(0.5, '#14532d');
            gradient.addColorStop(1, '#0369a1');
            context.fillStyle = gradient;
            context.fillRect(0, 0, width, height);

            const stripeWidth = Math.max(80, width / 5);
            const stripeX = ((state.frame * 7) % (width + stripeWidth)) - stripeWidth;
            context.fillStyle = 'rgba(255,255,255,0.18)';
            context.fillRect(stripeX, 0, stripeWidth, height);

            context.fillStyle = '#f8fafc';
            context.font = `${Math.max(24, Math.round(width / 22))}px Arial, sans-serif`;
            context.fillText('Geti stream resize check', Math.max(24, width * 0.06), Math.max(48, height * 0.16));
            context.font = `${Math.max(18, Math.round(width / 34))}px Arial, sans-serif`;
            context.fillText(state.label, Math.max(24, width * 0.06), Math.max(86, height * 0.25));

            context.strokeStyle = '#fbbf24';
            context.lineWidth = Math.max(6, width / 100);
            context.strokeRect(width * 0.08, height * 0.35, width * 0.34, height * 0.38);

            context.strokeStyle = '#38bdf8';
            context.strokeRect(width * 0.55, height * 0.2, width * 0.28, height * 0.52);

            requestAnimationFrame(draw);
        };

        canvas.width = state.width;
        canvas.height = state.height;
        draw();

        if (typeof canvas.captureStream !== 'function') {
            throw new Error('Synthetic stream recording requires canvas.captureStream support.');
        }

        state.stream = canvas.captureStream(30);
        window.__getiSyntheticStreamState = state;
        window.__getiResizeSyntheticStream = (width, height, label) => {
            state.width = width;
            state.height = height;
            state.label = label ?? `stream resized ${width} x ${height}`;
        };

        return state.stream;
    };

    class FakeRTCPeerConnection {
        constructor() {
            this.connectionState = 'new';
            this.iceGatheringState = 'complete';
            this.localDescription = undefined;
            this.remoteDescription = undefined;
            this.listeners = {};
            this.stream = ensureSyntheticStream();
            this.receiver = { track: this.stream.getVideoTracks()[0] };
            this.transceivers = [];
        }

        addTransceiver(kind, options = {}) {
            const transceiver = {
                kind,
                direction: options.direction ?? 'sendrecv',
                stop() {},
            };
            this.transceivers.push(transceiver);
            return transceiver;
        }

        createDataChannel() {
            const channel = {
                readyState: 'open',
                onopen: null,
                send() {},
                close() {},
                addEventListener() {},
                removeEventListener() {},
            };

            setTimeout(() => channel.onopen?.({ type: 'open' }), 20);
            return channel;
        }

        async createOffer() {
            return { type: 'offer', sdp: 'v=0\r\n' };
        }

        async setLocalDescription(description) {
            this.localDescription = description;
        }

        async setRemoteDescription(description) {
            this.remoteDescription = description;
            setTimeout(() => {
                this.connectionState = 'connected';
                this.emit('connectionstatechange');
                this.emit('track', { track: this.receiver.track, streams: [this.stream] });
            }, 650);
        }

        getReceivers() {
            return [this.receiver];
        }

        getTransceivers() {
            return this.transceivers;
        }

        getSenders() {
            return [];
        }

        close() {
            this.connectionState = 'closed';
            this.emit('connectionstatechange');
        }

        addEventListener(type, listener) {
            this.listeners[type] ??= new Set();
            this.listeners[type].add(listener);
        }

        removeEventListener(type, listener) {
            this.listeners[type]?.delete(listener);
        }

        emit(type, init = {}) {
            const event = { type, target: this, currentTarget: this, ...init };
            this.listeners[type]?.forEach((listener) => listener(event));
        }
    }

    window.RTCPeerConnection = FakeRTCPeerConnection;
    window.webkitRTCPeerConnection = FakeRTCPeerConnection;
};

const rectToObject = (rect) => ({
    x: Number(rect.x.toFixed(2)),
    y: Number(rect.y.toFixed(2)),
    width: Number(rect.width.toFixed(2)),
    height: Number(rect.height.toFixed(2)),
    right: Number(rect.right.toFixed(2)),
    bottom: Number(rect.bottom.toFixed(2)),
});

const main = async () => {
    await fs.mkdir(videoDir, { recursive: true });
    await fs.rm(finalVideoPath, { force: true });
    await fs.rm(measurementsPath, { force: true });
    await fs.rm(screenshotPath, { force: true });

    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
        viewport: { width: 1280, height: 720 },
        recordVideo: {
            dir: videoDir,
            size: { width: 1280, height: 720 },
        },
    });

    const page = await context.newPage();
    const consoleMessages = [];
    const pageErrors = [];

    page.on('console', (message) => {
        const text = message.text();
        consoleMessages.push({ type: message.type(), text });
        if (message.type() === 'error') {
            console.error(`[browser:${message.type()}] ${text}`);
        }
    });
    page.on('pageerror', (error) => {
        pageErrors.push(error.message);
        console.error(`[pageerror] ${error.message}`);
    });

    await page.addInitScript(syntheticWebRTCScript);

    await page.route('**/health', (route) => fulfillJson(route, { status: 'ok' }));
    await page.route('**/api/**', (route) => {
        const request = route.request();
        const url = new URL(request.url());
        const response = apiResponseFor(url.pathname, request.method());

        if (response.status === 204) {
            return route.fulfill({ status: 204 });
        }

        return fulfillJson(route, response.data, response.status ?? 200);
    });

    const measurements = [];
    const measure = async (label) => {
        const measurement = await page.evaluate((currentLabel) => {
            const video = document.querySelector('video');
            const transform = document.querySelector('[data-testid="zoom-transform"]');
            const wrapper = transform?.parentElement;

            if (!video || !transform || !wrapper) {
                return { label: currentLabel, missing: true };
            }

            const videoRect = video.getBoundingClientRect();
            const wrapperRect = wrapper.getBoundingClientRect();
            const transformStyle = getComputedStyle(transform).transform;
            const fitsContainer =
                videoRect.width <= wrapperRect.width + 2 &&
                videoRect.height <= wrapperRect.height + 2 &&
                videoRect.left >= wrapperRect.left - 2 &&
                videoRect.top >= wrapperRect.top - 2 &&
                videoRect.right <= wrapperRect.right + 2 &&
                videoRect.bottom <= wrapperRect.bottom + 2;

            return {
                label: currentLabel,
                viewport: { width: window.innerWidth, height: window.innerHeight },
                videoIntrinsic: { width: video.videoWidth, height: video.videoHeight },
                videoAttribute: {
                    width: Number(video.getAttribute('width')),
                    height: Number(video.getAttribute('height')),
                },
                wrapperRect: {
                    x: Number(wrapperRect.x.toFixed(2)),
                    y: Number(wrapperRect.y.toFixed(2)),
                    width: Number(wrapperRect.width.toFixed(2)),
                    height: Number(wrapperRect.height.toFixed(2)),
                    right: Number(wrapperRect.right.toFixed(2)),
                    bottom: Number(wrapperRect.bottom.toFixed(2)),
                },
                videoRect: {
                    x: Number(videoRect.x.toFixed(2)),
                    y: Number(videoRect.y.toFixed(2)),
                    width: Number(videoRect.width.toFixed(2)),
                    height: Number(videoRect.height.toFixed(2)),
                    right: Number(videoRect.right.toFixed(2)),
                    bottom: Number(videoRect.bottom.toFixed(2)),
                },
                transform: transformStyle,
                fitsContainer,
            };
        }, label);

        measurements.push(measurement);
        return measurement;
    };

    await page.goto(`${baseURL}/projects/${project.id}/inference`, { waitUntil: 'networkidle' });
    await page.getByRole('button', { name: 'Start stream' }).click();
    await page.waitForSelector('video');
    await page.waitForFunction(() => {
        const video = document.querySelector('video');
        return video && video.videoWidth === 640 && video.videoHeight === 480 && video.readyState >= 2;
    });
    await page.waitForTimeout(1200);
    await measure('initial-load-640x480');

    await page.evaluate(() => window.__getiResizeSyntheticStream?.(1280, 720, 'stream resized 1280 x 720'));
    await page.waitForFunction(() => {
        const video = document.querySelector('video');
        return video && video.videoWidth === 1280 && video.videoHeight === 720;
    });
    await page.waitForTimeout(1200);
    await measure('stream-resolution-resize-1280x720');

    await page.setViewportSize({ width: 960, height: 720 });
    await page.waitForTimeout(1300);
    await measure('browser-resize-960x720');

    await page.setViewportSize({ width: 1440, height: 900 });
    await page.waitForTimeout(1300);
    await measure('browser-resize-1440x900');

    await page.screenshot({ path: screenshotPath, fullPage: true });

    const missing = measurements.filter((entry) => entry.missing);
    const notFitting = measurements.filter((entry) => entry.fitsContainer === false);
    const viewportWidths = measurements.map((entry) => entry.wrapperRect?.width).filter(Boolean);
    const videoWidths = measurements.map((entry) => entry.videoRect?.width).filter(Boolean);
    const resizedWithViewport = new Set(viewportWidths).size > 1 && new Set(videoWidths).size > 1;
    const streamResizeCaptured = measurements.some(
        (entry) => entry.videoIntrinsic?.width === 1280 && entry.videoIntrinsic?.height === 720
    );

    const videoArtifact = page.video();
    await context.close();
    await browser.close();

    const recordedVideoPath = await videoArtifact.path();
    await fs.rename(recordedVideoPath, finalVideoPath);

    const summary = {
        passed: missing.length === 0 && notFitting.length === 0 && resizedWithViewport && streamResizeCaptured,
        finalVideoPath,
        screenshotPath,
        measurements,
        consoleErrors: consoleMessages.filter((message) => message.type === 'error'),
        pageErrors,
        checks: {
            allMeasurementsPresent: missing.length === 0,
            allFramesFitContainer: notFitting.length === 0,
            videoDimensionsChangedWithViewport: resizedWithViewport,
            streamResolutionResizeCaptured: streamResizeCaptured,
        },
    };

    await fs.writeFile(measurementsPath, JSON.stringify(summary, null, 2));
    console.log(JSON.stringify(summary, null, 2));

    if (!summary.passed) {
        process.exitCode = 1;
    }
};

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
