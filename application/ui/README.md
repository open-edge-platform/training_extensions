# Geti Tune UI

Modern React application for AI model training and inference, built with Rsbuild and TypeScript.

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
src/
├── api/              # API client (openapi-fetch) & type definitions
├── features/         # Feature modules (domain-driven)
│   ├── annotator/    # Annotation tools, canvas, providers
│   ├── dataset/      # Data collection, gallery, media preview
│   ├── inference/    # Live inference, WebRTC streaming
│   └── project/      # Project creation, configuration
├── routes/           # Page-level components
├── components/       # Shared UI components (zoom, etc.)
├── hooks/            # Custom React hooks
├── constants/        # App-wide constants (paths, config)
└── providers.tsx     # Global context providers
```

### Key Directories

**`features/`** - Domain-driven modules with their own components, hooks, and providers  
**`api/`** - Auto-generated TypeScript types from OpenAPI spec  
**`components/`** - Reusable UI primitives (not feature-specific)  
**`packages/`** - Shared UI library from [Geti](https://github.com/open-edge-platform/geti)

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
