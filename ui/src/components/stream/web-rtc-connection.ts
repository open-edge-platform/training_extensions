import { fetchClient } from '../../api/client';

export type WebRTCConnectionStatus = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'failed';

type WebRTCConnectionEvent =
    | {
          type: 'status_change';
          status: WebRTCConnectionStatus;
      }
    | {
          type: 'error';
          error: Error;
      };

type Listener = (event: WebRTCConnectionEvent) => void;

export class WebRTCConnection {
    private peerConnection: RTCPeerConnection | undefined;
    private webrtcId: string;
    private status: WebRTCConnectionStatus = 'idle';
    private dataChannel: RTCDataChannel | undefined;

    private listeners: Array<Listener> = [];

    constructor() {
        this.webrtcId = Math.random().toString(36).substring(7);
    }

    public getStatus(): WebRTCConnectionStatus {
        return this.status;
    }

    public getPeerConnection(): RTCPeerConnection | undefined {
        return this.peerConnection;
    }

    public getId(): string {
        return this.webrtcId;
    }

    public async start(): Promise<void> {
        if (this.peerConnection && this.status !== 'idle' && this.status !== 'disconnected') {
            console.warn('WebRTC connection is already active or in progress.');
            return;
        }

        this.updateStatus('connecting');
        this.peerConnection = new RTCPeerConnection();

        const timeoutId = setTimeout(() => {
            console.warn('Connection is taking longer than usual. Are you on a VPN?');
        }, 5000);

        try {
            this.peerConnection.addTransceiver('video', { direction: 'recvonly' });

            this.dataChannel = this.peerConnection.createDataChannel('text');
            this.dataChannel.onopen = () => {
                this.dataChannel?.send('handshake');
            };

            const offer = await this.peerConnection.createOffer();
            await this.peerConnection.setLocalDescription(offer);

            await new Promise<void>((resolve) => {
                if (!this.peerConnection) {
                    resolve();
                    return;
                }
                if (this.peerConnection.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    const checkState = () => {
                        if (this.peerConnection && this.peerConnection.iceGatheringState === 'complete') {
                            this.peerConnection.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    };
                    this.peerConnection.addEventListener('icegatheringstatechange', checkState);
                }
            });

            const response = await fetchClient.POST('/api/webrtc/offer', {
                body: {
                    sdp: this.peerConnection.localDescription?.sdp,
                    type: this.peerConnection.localDescription?.type ?? '',
                    webrtc_id: this.webrtcId,
                },
            });

            const data = response.data as
                | RTCSessionDescriptionInit
                | {
                      status: 'failed';
                      meta: { error: 'concurrency_limit_reached'; limit: number };
                  };

            if ('status' in data && data.status === 'failed') {
                const errorMessage =
                    data.meta.error === 'concurrency_limit_reached'
                        ? `Too many connections. Maximum limit is ${data.meta.limit}`
                        : data.meta.error;
                console.error(errorMessage);

                this.emit({ type: 'error', error: new Error(errorMessage) });
                return;
            }

            await this.peerConnection.setRemoteDescription(data as RTCSessionDescriptionInit);

            await this.updateConfThreshold(0.5); // Initial confidence threshold

            this.peerConnection.addEventListener('connectionstatechange', () => {
                if (!this.peerConnection) return;
                switch (this.peerConnection.connectionState) {
                    case 'connected':
                        this.updateStatus('connected');
                        clearTimeout(timeoutId);
                        break;
                    case 'disconnected':
                        this.updateStatus('disconnected');
                        break;
                    case 'failed':
                        this.updateStatus('failed');
                        this.emit({ type: 'error', error: new Error('WebRTC connection failed.') });
                        break;
                    case 'closed':
                        this.updateStatus('disconnected');
                        break;
                    default:
                        // 'new', 'connecting'
                        this.updateStatus('connecting');
                        break;
                }
            });
        } catch (err) {
            clearTimeout(timeoutId);
            console.error('Error setting up WebRTC:', err);
            this.emit({ type: 'error', error: err as Error });
            this.updateStatus('failed');
            this.stop();
        }

        if (this.peerConnection) {
            this.peerConnection.getTransceivers().forEach((t) => (t.direction = 'recvonly'));
        }
    }

    public async stop(): Promise<void> {
        if (!this.peerConnection) {
            return;
        }

        if (this.peerConnection.getTransceivers) {
            this.peerConnection.getTransceivers().forEach((transceiver) => {
                if (transceiver.stop) {
                    transceiver.stop();
                }
            });
        }

        if (this.peerConnection.getSenders) {
            this.peerConnection.getSenders().forEach((sender) => {
                if (sender.track && sender.track.stop) sender.track.stop();
            });
        }

        // Give a brief moment for tracks to stop before closing the connection
        await new Promise<void>((resolve) =>
            setTimeout(() => {
                if (this.peerConnection) {
                    this.peerConnection.close();
                    this.peerConnection = undefined;
                    this.updateStatus('idle');
                }

                resolve();
            }, 500)
        );
    }

    public subscribe(listener: Listener): () => void {
        this.listeners.push(listener);

        return () => this.off(listener);
    }

    private off(listener: Listener) {
        this.listeners = this.listeners.filter((l) => l !== listener);
    }

    private emit(event: WebRTCConnectionEvent) {
        this.listeners.forEach((listener) => listener(event));
    }

    private updateStatus(newStatus: WebRTCConnectionStatus) {
        if (this.status !== newStatus) {
            this.status = newStatus;
            this.emit({ type: 'status_change', status: newStatus });
        }
    }

    private updateConfThreshold(value: number) {
        return fetchClient.POST('/api/input_hook', {
            body: {
                conf_threshold: value,
                webrtc_id: this.webrtcId,
            },
        });
    }
}
