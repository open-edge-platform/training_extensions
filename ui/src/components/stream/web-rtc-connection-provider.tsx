import { createContext, ReactNode, useContext, useEffect, useRef, useState } from 'react';

import { WebRTCConnection, WebRTCConnectionStatus } from './web-rtc-connection';

type WebRTCConnectionContextType = {
    status: WebRTCConnectionStatus;
    peerConnection: RTCPeerConnection | undefined;
    start: () => Promise<void>;
    stop: () => Promise<void>;
};

const WebRTCConnectionContext = createContext<WebRTCConnectionContextType | null>(null);

export const useWebRTCConnectionState = (ConnectionClass: new () => WebRTCConnection = WebRTCConnection) => {
    const webRTCConnectionRef = useRef<WebRTCConnection | null>(null);
    const [status, setStatus] = useState<WebRTCConnectionStatus>('idle');
    const [peerConnection, setPeerConnection] = useState<RTCPeerConnection | undefined>(undefined);

    useEffect(() => {
        if (!webRTCConnectionRef.current) {
            webRTCConnectionRef.current = new ConnectionClass();
        }

        const webRTCConnection = webRTCConnectionRef.current;

        setStatus(webRTCConnection.getStatus());
        setPeerConnection(webRTCConnection.getPeerConnection());

        const unsubscribe = webRTCConnection.subscribe((event) => {
            if (event.type === 'status_change') {
                setStatus(event.status);
                setPeerConnection(webRTCConnection.getPeerConnection());
            }

            if (event.type === 'error') {
                console.error('WebRTC Connection Error:', event.error);
            }
        });

        return () => {
            unsubscribe();
            webRTCConnection.stop();
            webRTCConnectionRef.current = null;
            setStatus('idle');
            setPeerConnection(undefined);
        };
    }, [ConnectionClass]);

    const start = async () => {
        if (!webRTCConnectionRef.current) return;

        try {
            await webRTCConnectionRef.current.start();
        } catch (error) {
            console.error('Failed to start WebRTC connection:', error);
        }
    };

    const stop = async () => {
        if (!webRTCConnectionRef.current) return;

        try {
            await webRTCConnectionRef.current.stop();
        } catch (error) {
            console.error('Failed to stop WebRTC connection:', error);
        }
    };

    return {
        status,
        peerConnection,
        start,
        stop,
    };
};

export const WebRTCConnectionProvider = ({ children }: { children: ReactNode }) => {
    const value = useWebRTCConnectionState(WebRTCConnection);

    return <WebRTCConnectionContext.Provider value={value}>{children}</WebRTCConnectionContext.Provider>;
};

export const useWebRTCConnection = () => {
    const context = useContext(WebRTCConnectionContext);

    if (context === null) {
        throw new Error('useWebRTCConnection was used outside of WebRTCConnectionProvider');
    }

    return context;
};
