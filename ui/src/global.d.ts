import 'react';

import { type NavigateOptions } from 'react-router-dom';

declare module 'react' {
    interface CSSProperties {
        [key: `--${string}`]: string | number;
    }
}

declare module '@adobe/react-spectrum' {
    interface RouterConfig {
        routerOptions: NavigateOptions;
    }
}
