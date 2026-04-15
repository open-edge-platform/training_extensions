// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useEffect, useRef } from 'react';

import { View } from '@geti/ui';

import { useWebRTCConnection } from './web-rtc-connection-provider';

import classes from './stream.module.scss';

const useStreamToVideo = () => {
    const videoRef = useRef<HTMLVideoElement>(null);

    const { status, webRTCConnectionRef } = useWebRTCConnection();

    const connect = useCallback(async () => {
        const videoOutput = videoRef.current;
        const webrtcConnection = webRTCConnectionRef.current;
        const peerConnection = webrtcConnection?.getPeerConnection();

        if (!peerConnection) {
            return;
        }

        const receivers = peerConnection.getReceivers() ?? [];
        const stream = new MediaStream(receivers.map((receiver) => receiver.track).filter(Boolean));

        if (videoOutput && videoOutput.srcObject !== stream) {
            videoOutput.srcObject = stream;
        }
    }, [videoRef, webRTCConnectionRef]);

    useEffect(() => {
        if (status === 'connected') {
            connect();
        }
    }, [status, connect]);

    useEffect(() => {
        const webrtcConnection = webRTCConnectionRef.current;
        const peerConnection = webrtcConnection?.getPeerConnection();

        if (!peerConnection) {
            return;
        }

        peerConnection.addEventListener('track', connect);

        return () => {
            peerConnection.removeEventListener('track', connect);
        };
    }, [webRTCConnectionRef, connect]);

    return videoRef;
};

export const Stream = () => {
    const videoRef = useStreamToVideo();
    const { status } = useWebRTCConnection();

    return (
        <View gridArea={'innercanvas'} width={'100%'} height={'100%'}>
            {status === 'connected' && (
                // eslint-disable-next-line jsx-a11y/media-has-caption
                <video ref={videoRef} autoPlay playsInline controls={false} className={classes.video} />
            )}
        </View>
    );
};
