import { ReactNode } from 'react';

import { ThemeProvider } from '@geti/ui/theme';
import { broadcastQueryClient } from '@tanstack/query-broadcast-client-experimental';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { WebRTCConnectionProvider } from './components/stream/web-rtc-connection-provider';
import { ZoomProvider } from './components/zoom/zoom';

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            gcTime: 30 * 60 * 1000,
            staleTime: 5 * 60 * 1000,
        },
        mutations: {
            onSuccess: async () => {
                await queryClient.invalidateQueries();
            },
        },
    },
});

// Sync our server state with all browser tabs
broadcastQueryClient({ queryClient, broadcastChannel: 'geti-edge' });

export function Providers({ children }: { children: ReactNode }) {
    return (
        <QueryClientProvider client={queryClient}>
            <ThemeProvider>
                <WebRTCConnectionProvider>
                    <ZoomProvider>{children}</ZoomProvider>
                </WebRTCConnectionProvider>
            </ThemeProvider>
        </QueryClientProvider>
    );
}
