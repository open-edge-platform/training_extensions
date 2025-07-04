import { createContext, ReactNode, RefObject, useCallback, useContext, useEffect, useRef, useState } from 'react';

import { WebRTCConnection, WebRTCConnectionStatus } from './web-rtc-connection'; // Adjust path

export type WebRTCConnectionState = null | {
    status: WebRTCConnectionStatus;
    start: () => Promise<void>;
    stop: () => Promise<void>;
    webrtcConnectionRef: RefObject<WebRTCConnection>;
};

export const WebRTCConnectionContext = createContext<WebRTCConnectionState>(null);

function useWebRTCConnectionState() {
    const webrtcConnectionRef = useRef<WebRTCConnection | null>(null);
    const [status, setStatus] = useState<WebRTCConnectionStatus>('idle');

    // Initialize WebRTCConnection on mount
    useEffect(() => {
        if (webrtcConnectionRef.current) {
            return;
        }

        const webRTCConnection = new WebRTCConnection();
        webrtcConnectionRef.current = webRTCConnection;

        const unsubscribe = webRTCConnection.subscribe((event) => {
            if (event.type === 'status_change') {
                setStatus(event.status);
            }

            if (event.type === 'error') {
                console.error('WebRTC Connection Error:', event.error);
                // Optionally update status to 'failed' if not already
                if (webrtcConnectionRef.current?.getStatus() !== 'failed') {
                    setStatus('failed');
                }
            }
        });

        return () => {
            unsubscribe();
            webRTCConnection.stop(); // Ensure connection is closed on unmount
            webrtcConnectionRef.current = null;
        };
    }, []);

    const start = useCallback(async () => {
        if (!webrtcConnectionRef.current) {
            return;
        }
        try {
            await webrtcConnectionRef.current.start();
        } catch (error) {
            console.error('Failed to start WebRTC connection:', error);
            setStatus('failed');
        }
    }, []);

    const stop = useCallback(async () => {
        if (!webrtcConnectionRef.current) {
            return;
        }

        await webrtcConnectionRef.current.stop();
    }, []);

    return {
        start,
        stop,
        status,
        webrtcConnectionRef,
    };
}

export function WebRTCConnectionProvider({ children }: { children: ReactNode }) {
    const value = useWebRTCConnectionState();

    return <WebRTCConnectionContext.Provider value={value}>{children}</WebRTCConnectionContext.Provider>;
}

export function useWebRTCConnection() {
    const context = useContext(WebRTCConnectionContext);

    if (context === null) {
        throw new Error('useWebRTCConnection was used outside of WebRTCConnectionProvider');
    }

    return context;
}
