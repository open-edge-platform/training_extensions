import { fireEvent, render } from '@testing-library/react';

import { Listener, WebRTCConnection, WebRTCConnectionStatus } from './web-rtc-connection';
import { useWebRTCConnectionState, WebRTCConnectionProvider } from './web-rtc-connection-provider';

class MockWebRTCConnection {
    status: WebRTCConnectionStatus = 'idle';
    listeners: Listener[] = [];
    getStatus() {
        return this.status;
    }
    getPeerConnection() {
        return undefined;
    }
    async start() {
        this.status = 'connected';
        this.listeners.forEach((l) => l({ type: 'status_change', status: this.status }));
    }
    async stop() {
        this.status = 'idle';
        this.listeners.forEach((l) => l({ type: 'status_change', status: this.status }));
    }
    subscribe(listener: Listener) {
        this.listeners.push(listener);
        return () => {
            this.listeners = this.listeners.filter((currentListener) => currentListener !== listener);
        };
    }
}

describe('WebRTCConnectionProvider', () => {
    // @ts-expect-error We only care about a few methods of the class
    const App = ({ customConnection = MockWebRTCConnection }: { customConnection?: new () => WebRTCConnection }) => {
        const { status, start, stop } = useWebRTCConnectionState(customConnection);

        return (
            <>
                <span aria-label='status'>{status}</span>
                <button aria-label='start' onClick={start}>
                    Start
                </button>
                <button aria-label='stop' onClick={stop}>
                    Stop
                </button>
            </>
        );
    };

    afterEach(() => {
        jest.clearAllMocks();
    });

    it('provides initial status as idle', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        expect(getByLabelText('status')).toHaveTextContent('idle');
    });

    it('updates status to connected after start', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));

        expect(getByLabelText('status')).toHaveTextContent('connected');
    });

    it('updates status to idle after stop', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));

        expect(getByLabelText('status')).toHaveTextContent('connected');

        fireEvent.click(getByLabelText('stop'));

        expect(getByLabelText('status')).toHaveTextContent('idle');
    });

    it('cleans up on unmount', () => {
        const stopSpy = jest.spyOn(MockWebRTCConnection.prototype, 'stop');

        const { unmount } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        unmount();

        expect(stopSpy).toHaveBeenCalled();
    });

    it('supports multiple consumers', () => {
        const App2 = ({
            // @ts-expect-error We only care about a few methods of the class
            customConnection = MockWebRTCConnection,
        }: {
            customConnection?: new () => WebRTCConnection;
        }) => {
            const { status, start } = useWebRTCConnectionState(customConnection);

            return (
                <>
                    <span aria-label='status2'>{status}</span>
                    <button aria-label='start2' onClick={start}>
                        Start
                    </button>
                </>
            );
        };

        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
                <App2 />
            </WebRTCConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));
        fireEvent.click(getByLabelText('start2'));

        expect(getByLabelText('status')).toHaveTextContent('connected');
        expect(getByLabelText('status2')).toHaveTextContent('connected');
    });

    it('handles status sequence: start -> stop -> start', () => {
        const { getByLabelText } = render(
            <WebRTCConnectionProvider>
                <App />
            </WebRTCConnectionProvider>
        );

        fireEvent.click(getByLabelText('start'));
        expect(getByLabelText('status')).toHaveTextContent('connected');

        fireEvent.click(getByLabelText('stop'));
        expect(getByLabelText('status')).toHaveTextContent('idle');

        fireEvent.click(getByLabelText('start'));
        expect(getByLabelText('status')).toHaveTextContent('connected');
    });
});
