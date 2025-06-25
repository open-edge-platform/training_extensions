import { Dispatch, RefObject, SetStateAction, useCallback, useEffect, useRef, useState } from 'react';

import { Button, ToggleButton } from '@geti/ui';

import { fetchClient } from '../../api/client';

// TODO: create an abstraction for this based on,
// https://github.com/node-webrtc/node-webrtc-examples/blob/master/lib/client/index.js
function useStream() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const peerConnectionRef = useRef<RTCPeerConnection>(undefined);
    const webrtc_id = useRef<string>(Math.random().toString(36).substring(7));

    const [status, setStatus] = useState<'idle' | 'running'>('idle');

    const stop = useCallback(function stop() {
        const peerConnection = peerConnectionRef.current!;
        const videoOutput = videoRef.current!;

        if (peerConnection) {
            if (peerConnection.getTransceivers) {
                peerConnection.getTransceivers().forEach((transceiver) => {
                    if (transceiver.stop) {
                        transceiver.stop();
                    }
                });
            }

            if (peerConnection.getSenders) {
                peerConnection.getSenders().forEach((sender) => {
                    if (sender.track && sender.track.stop) sender.track.stop();
                });
            }

            setTimeout(() => {
                peerConnection.close();
            }, 500);
        }

        videoOutput.srcObject = null;
        setStatus('idle');
    }, []);

    const start = useCallback(
        async function setupWebRTC() {
            if (peerConnectionRef.current === undefined) {
                peerConnectionRef.current = new RTCPeerConnection();
            }

            const peerConnection = peerConnectionRef.current!;
            const videoOutput = videoRef.current!;

            function updateConfThreshold(value: number) {
                fetchClient.POST('/api/input_hook', {
                    body: {
                        conf_threshold: value,
                        webrtc_id: webrtc_id.current,
                    },
                });
            }

            const timeoutId = setTimeout(() => {
                console.warn('Connection is taking longer than usual. Are you on a VPN?');
            }, 5000);

            try {
                peerConnection.addEventListener('track', (evt) => {
                    if (videoOutput && videoOutput.srcObject !== evt.streams[0]) {
                        videoOutput.srcObject = evt.streams[0];
                    }
                });

                //const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                //stream.getTracks().forEach((track) => {
                //    peerConnection.addTrack(track, stream);
                //});
                peerConnection.addTransceiver('video', { direction: 'recvonly' });

                // TODO: we can remove these as we don't rely on this info
                const dataChannel = peerConnection.createDataChannel('text');
                dataChannel.onopen = () => {
                    console.info('Data channel is open');
                    dataChannel.send('handshake');
                };
                dataChannel.onmessage = (event) => {
                    const eventJson = JSON.parse(event.data);
                    if (eventJson.type === 'send_input') {
                        updateConfThreshold(0.4);
                    }
                };

                //
                const offer = await peerConnection.createOffer();
                await peerConnection.setLocalDescription(offer);

                await new Promise<void>((resolve) => {
                    if (peerConnection.iceGatheringState === 'complete') {
                        resolve();
                    } else {
                        const checkState = () => {
                            if (peerConnection.iceGatheringState === 'complete') {
                                peerConnection.removeEventListener('icegatheringstatechange', checkState);
                                resolve();
                            }
                        };
                        peerConnection.addEventListener('icegatheringstatechange', checkState);
                    }
                });

                const response = await fetchClient.POST('/api/webrtc/offer', {
                    body: {
                        sdp: peerConnection.localDescription?.sdp,
                        type: peerConnection.localDescription?.type ?? '',
                        webrtc_id: webrtc_id.current,
                    },
                });

                const data = response.data as
                    | RTCSessionDescriptionInit
                    | {
                          status: 'failed';
                          meta: { error: 'concurrency_limit_reached'; limit: number };
                      };

                if ('status' in data && data.status === 'failed') {
                    console.error(
                        data.meta.error === 'concurrency_limit_reached'
                            ? `Too many connections. Maximum limit is ${data.meta.limit}`
                            : data.meta.error
                    );
                    stop();
                    return;
                }

                await peerConnection.setRemoteDescription(data as RTCSessionDescriptionInit);

                // Send initial confidence threshold
                updateConfThreshold(0.5);

                peerConnection.addEventListener('connectionstatechange', () => {
                    if (peerConnection.connectionState === 'connected') {
                        clearTimeout(timeoutId);
                        setStatus('running');
                    }
                });
            } catch (err) {
                clearTimeout(timeoutId);
                console.error('Error setting up WebRTC:', err);
                stop();
            }

            peerConnection.getTransceivers().forEach((t) => (t.direction = 'recvonly'));
        },
        [stop]
    );

    return { start, stop, videoRef, status };
}

function useVideoSize(
    setSize: Dispatch<SetStateAction<{ width: number; height: number }>>,
    videoRef: RefObject<HTMLVideoElement | null>
) {
    useEffect(() => {
        const video = videoRef.current;

        const onsize = video?.addEventListener('loadedmetadata', (event) => {
            const target = event.currentTarget as HTMLVideoElement;
            if (target.videoWidth && target.videoHeight) {
                setSize({ width: target.videoWidth, height: target.videoHeight });
            }
        });

        const onresize = video?.addEventListener('resize', (event) => {
            const target = event.currentTarget as HTMLVideoElement;
            if (target.videoWidth && target.videoHeight) {
                setSize({ width: target.videoWidth, height: target.videoHeight });
            }
        });

        return () => {
            if (onsize) {
                video?.removeEventListener('loadedmetadata', onsize);
            }
            if (onresize) {
                video?.removeEventListener('resize', onresize);
            }
        };
    }, [setSize, videoRef]);
}

export function Stream({
    size,
    setSize,
}: {
    size: { width: number; height: number };
    setSize: Dispatch<SetStateAction<{ width: number; height: number }>>;
}) {
    const { start, stop, videoRef } = useStream();

    const { width, height } = size;
    useVideoSize(setSize, videoRef);
    const [isFocussed, setIsFocussed] = useState(false);

    return (
        <>
            <div style={{ gridArea: 'innercanvas' }}>
                <div
                    style={{
                        position: 'absolute',
                        top: '20px',
                        left: '20px',
                        zIndex: 10,
                        display: 'flex',
                        gap: '1rem',
                    }}
                >
                    <Button onPress={start}>Start</Button>
                    <Button onPress={stop}>Stop</Button>
                    <ToggleButton onChange={(value) => setIsFocussed(value)} isSelected={isFocussed} isEmphasized>
                        Focus
                    </ToggleButton>
                </div>
                {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    width={width}
                    height={height}
                    controls={false}
                    style={{
                        background: 'var(--spectrum-global-color-gray-200)',
                    }}
                />
            </div>
        </>
    );
}
