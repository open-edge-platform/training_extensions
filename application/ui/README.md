# Geti Tune UI

Modern React application for AI model training and inference, built with Rsbuild and TypeScript.

The Geti Edge applications aim to provide a user experience and design language consistent with the main [Geti application](https://github.com/open-edge-platform/geti). To achieve this, we reuse many architectural decisions from Geti.

## Goals

- **Developer Experience**: Setting up a developer environment for both the UI and server should take only seconds.
- **API Adaptability**: Adapting the UI to REST API changes should require minimal effort.
- **Consistency**: Maintain a unified look, feel, and user experience across the whole Geti ecosystem through shared design language, architectural patterns, and reusable components.

## 🚀 Quick Start

### Prerequisites

- Node.js >= v24.2.0
- npm >= v11.3.0

### Installation

```bash
npm install
```

### Development

```bash
npm run dev          # Start dev server at localhost:3000
npm run server       # Start backend at localhost:7860 (separate terminal)
```

### Build

```bash
npm run build        # Production build
npm run preview      # Preview production build
```

## 📁 Project Structure

```
.
├── packages/
│   ├── config           # Shared configuration (`@geti/config`)
│   ├── smart-tools      # AI algorithms and tools (`@geti/smart-tools`)
│   └── ui               # Shared UI library (`@geti/ui`)
├── src/
│   ├── api/             # OpenAPI client and query hooks
│   ├── assets/          # Images, illustrations, icons
│   ├── components/      # Reusable UI components
│   ├── features/        # Application-specific feature modules
│   │   ├── annotator/   # Annotation tools, canvas, providers
│   │   ├── dataset/     # Data collection, gallery, media preview
│   │   ├── inference/   # Live inference, WebRTC streaming
│   │   └── project/     # Project creation, configuration
│   ├── hooks/           # Common hooks not necessarily related to a feature
│   ├── routes/          # Route elements and loaders
│   ├── constants/       # App-wide constants (paths, config)
│   ├── providers.tsx    # Global providers (QueryClient, Theme, Router, etc.)
│   └── router.tsx       # Application entrypoint (routing setup)
├── src-tauri/           # Tauri configuration
└── tests/               # Component and E2E tests
```

### Key Directories

**`api/`** - OpenAPI client and tanstack/query hooks, auto-generated TypeScript types from OpenAPI spec  
**`components/`** - Locally reusable UI primitives; promote mature components to `@geti/ui` (use `.component.tsx` suffix)  
**`features/`** - Domain-driven modules with their own components, hooks, and providers  
**`routes/`** & **`router.tsx`** - Routing setup; keep route files minimal—extract complex routes into feature modules  
**`providers.tsx`** - Global application providers (e.g., `QueryClientProvider`, `ThemeProvider`, `RouterProvider`)  
**`packages/`** - Shared libraries from Geti ecosystem

> **Note:**  
> Currently, `@geti/ui`, `@geti/config`, and `@geti/smart-tools` are installed via [Degit](https://github.com/Rich-Harris/degit) to avoid npm publishing. These will be published to npm as the Geti Edge ecosystem matures.

## 🔧 Common Tasks

### Update API Types

```bash
npm run update-spec  # Fetch OpenAPI spec & generate types
```

### Testing

```bash
npm run test:unit           # Unit tests (Vitest)
npm run test:unit:watch     # Watch mode
npm run test:e2e            # E2E tests (Playwright)
npm run test:component      # Component tests (Playwright)
```

### Code Quality

```bash
npm run lint          # Check for issues
npm run lint:fix      # Auto-fix issues
npm run format        # Format with Prettier
npm run cyclic-deps-check  # Detect circular dependencies
```

## 🏗️ Architecture Patterns

### Core Pillars

The application is built on four main pillars:

1. **Build System**: Modern web tooling for React-based applications
2. **Application Architecture**: Type-safe API integration with minimal effort
3. **Testing & CI/CD**: Robust testing setup that works locally and in CI
4. **AI Algorithms**: Interactive AI with low-latency algorithms

#### Build System

- **React** - Component-based UI architecture
- **TypeScript** - Static typing for reliability and maintainability
- **Rsbuild** - Fast and robust build toolchain for bundling, optimization, and environment targeting
- **Tauri** - Cross-platform desktop app packaging
- **ESLint & Prettier** - Enforced via `@geti/config` for code consistency and best practices

#### Application Architecture

- **React Router** - Single-page application navigation with dynamic routing
- **Tanstack Query** & [`openapi-react-query`](https://openapi-ts.dev/openapi-react-query/) - Server state management and type-safe API consumption
- **@geti/ui** - Shared visual components for consistent UX
- **React Context & Local State** - Local state via `useState`, shared state via `createContext`

#### Testing & CI/CD

- **Vitest** - Fast unit and integration testing
- **Playwright** - Component and end-to-end testing with shared MSW configuration for auto-mocking REST endpoints
- **Testing Library** - User-centric React component testing; follow [guiding principles](https://testing-library.com/docs/guiding-principles) for accessibility
- **MSW + OpenAPI** - Mock Service Worker with OpenAPI specs to simulate API responses in tests
- **GitHub Actions** - Automated CI/CD pipelines for building, testing, and deploying

#### Algorithms & AI

- **@geti/smart-tools** - Suite of intelligent tools for advanced functionality and optimization
- **WebRTC API** - Live video feeds with prediction overlays
- **WebAssembly** - High-performance, browser-executed code for compute-intensive tasks
- **OpenCV** - Image processing and computer vision
- **ONNXRuntime** - In-browser machine learning model execution for predictive analytics

### Feature Structure

Each feature follows a consistent pattern:

```
feature-name/
├── components/       # Feature-specific components
├── hooks/           # Feature-specific hooks
├── providers/       # Context providers
├── types.ts         # TypeScript interfaces
└── utils.ts         # Helper functions
```

### State Management

- **Context Providers** for domain state (annotations, zoom, visibility)
- **React Query** for server state (via `$api`)
- Local state with `useState` for UI-only concerns

### Code Conventions

- Functional components with TypeScript
- Custom hooks prefixed with `use*`
- Providers suffixed with `*Provider`
- Components use `.component.tsx` extension

## 🤝 Contributing

1. **Branch naming**: `your-username/fix-issue-1`
2. **Commit format**: Follow conventional commits
3. **Before PR**:
    ```bash
    npm run lint:fix
    npm run format
    npm run test:unit
    npm run test:component
    npm run build
    ```
4. **Keep changes focused** - one fix per PR

### Adding a New Feature

1. Create folder in `src/features/`
2. Add components, hooks, providers as needed
3. Update routes in `router.tsx`
4. Add tests

## 🔗 API Integration

The app uses `openapi-fetch` with auto-generated types:

```tsx
import { $api } from '@/api/client';

// Query
const { data } = $api.useQuery('get', '/api/projects');

// Mutation
const mutation = $api.useMutation('post', '/api/projects');
mutation.mutate({ body: { name: 'Project' } });
```

## 🖥️ Desktop App (WIP)

```bash
npm run start:desktop
```

Built with [Tauri](https://tauri.app) for native desktop experience.

---

For backend setup, see `../backend/readme.md`
