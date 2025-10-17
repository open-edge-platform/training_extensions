# ADR 005: Feature-Based Architecture

## Context

We needed a scalable folder structure that:

- Supports growth from small to large application
- Makes feature boundaries explicit
- Reduces cognitive load when navigating codebase
- Avoids deeply nested folder hierarchies
- Enables team autonomy (multiple devs working on different features)

Traditional MVC/layered architectures become hard to navigate as apps grow:

```
❌ Hard to scale
src/
├── components/        # 100+ components mixed together
├── hooks/            # Which hook belongs to which feature?
├── services/         # Tightly coupled services
└── utils/            # Grab-bag of utilities
```

## Decision

Use a **feature-based architecture** where code is organized by domain, not by type:

```
✅ Scales well
src/
├── features/          # Domain features
│   ├── annotator/    # Everything for annotation
│   ├── dataset/      # Everything for dataset management
│   ├── inference/    # Everything for inference
│   └── project/      # Everything for project management
├── components/        # Truly shared components
├── hooks/            # Truly shared hooks
├── api/              # API client (shared)
└── constants/        # Global constants
```

## Feature Structure

Each feature is a **self-contained module** with its own:

```
features/annotator/
├── index.ts                       # Public API (exports)
├── annotator.provider.tsx         # Feature-level state
├── annotator.routes.tsx           # Feature routes
├── components/
│   ├── canvas/
│   │   ├── canvas.component.tsx
│   │   ├── canvas.component.scss
│   │   └── canvas.test.tsx
│   ├── toolbar/
│   ├── shapes/
│   └── labels/
├── hooks/
│   ├── use-annotation-actions.ts
│   ├── use-draw-shape.ts
│   └── use-select-annotation.ts
├── providers/
│   ├── annotation-actions-provider.tsx
│   ├── annotation-visibility-provider.tsx
│   └── zoom-provider.tsx
├── utils/
│   ├── shape-calculations.ts
│   ├── coordinate-transform.ts
│   └── polylabel.ts
├── types/
│   ├── annotation.ts
│   └── shape.ts
└── constants/
    ├── tools.ts
    └── colors.ts
```

## Implementation

### Feature Public API

```typescript
// features/annotator/index.ts
// Only export what other features need

export { AnnotatorProvider } from './annotator.provider';
export { AnnotatorRoutes } from './annotator.routes';
export { useAnnotationActions } from './hooks/use-annotation-actions';
export type { Annotation, Shape } from './types';

// Internal components NOT exported
// - Canvas, Toolbar, Labels, etc.
```

### Feature Routes

```typescript
// features/annotator/annotator.routes.tsx
import { Route } from 'react-router-dom';
import { Canvas } from './components/canvas/canvas.component';

export const AnnotatorRoutes = () => (
    <>
        <Route path="/projects/:id/annotate" element={<Canvas />} />
    </>
);
```

### Root Router

```typescript
// src/router.tsx
import { AnnotatorRoutes } from './features/annotator';
import { DatasetRoutes } from './features/dataset';
import { ProjectRoutes } from './features/project';

export const Router = () => (
    <BrowserRouter>
        <Routes>
            {AnnotatorRoutes()}
            {DatasetRoutes()}
            {ProjectRoutes()}
        </Routes>
    </BrowserRouter>
);
```

## Feature Boundaries

### ✅ Good - Features Are Independent

```typescript
// features/dataset/components/media-gallery.tsx

import { fetchMediaItems } from '@/api/media';
import { Button } from '@geti/ui';
import { useQuery } from '@tanstack/react-query';

// Only imports from:
// 1. External libraries
// 2. Shared UI components
// 3. API client
// NO imports from other features
```

### ⚠️ Acceptable - Features Communicate via API

```typescript
// features/annotator/components/canvas.tsx

import { fetchAnnotations } from '@/api/annotations';
import { useQuery } from '@tanstack/react-query';

// Features don't talk to each other directly
// They talk via API (single source of truth)
```

### ❌ Bad - Features Depend on Each Other

```typescript
// features/annotator/components/canvas.tsx

import { MediaGallery } from '@/features/dataset/components/media-gallery';

// ❌ Creates tight coupling between features
```

## Shared Code Guidelines

### When to Share

Code belongs in `/src/components` or `/src/hooks` if:

1. **Used by 3+ features** (not just 2)
2. **Generic and reusable** (not domain-specific)
3. **Stable interface** (won't change per feature needs)

```typescript
// ✅ Truly shared - Used by annotator, dataset, inference
src / components / loading -
    spinner /
        // ✅ Truly shared - Used everywhere
        src /
        hooks /
        use -
    debounce.ts;

// ❌ Not shared - Only used in annotator
features / annotator / hooks / use - draw - rectangle.ts;
```

### Shared Components

```
src/components/
├── error-boundary/        # Used everywhere for error handling
├── empty-state/          # Used in dataset, inference, project
└── confirmation-dialog/  # Used in dataset, project
```

### Feature-Specific Components

```
features/annotator/components/
├── canvas/               # Only annotator needs canvas
├── shape-editor/         # Only annotator needs shape editing
└── polygon-tool/         # Only annotator needs polygon tool
```

## Dependency Rules

```
┌─────────────────┐
│   Features      │ ← Can import from shared
│  (annotator)    │ ← Can import from api/
│                 │ ← Can import from @geti/ui
└─────────────────┘ ← CANNOT import from other features
         ↓
┌─────────────────┐
│  Shared Code    │ ← Can import from api/
│ (components/)   │ ← Can import from @geti/ui
│                 │ ← CANNOT import from features/
└─────────────────┘
         ↓
┌─────────────────┐
│   API Client    │ ← Can only import from openapi-fetch
│    (api/)       │ ← CANNOT import from features or components
└─────────────────┘
```

## Feature Communication Patterns

### 1. Via URL Parameters

```typescript
// features/dataset/components/media-gallery.tsx
const navigate = useNavigate();

const handleAnnotate = (mediaId: string) => {
    navigate(`/projects/${projectId}/annotate?media=${mediaId}`);
};

// features/annotator/components/canvas.tsx
const [searchParams] = useSearchParams();
const mediaId = searchParams.get('media');
```

### 2. Via Shared API State (React Query)

```typescript
// Both features read from same cache
const { data: project } = useQuery({
    queryKey: ['projects', projectId],
    queryFn: () => fetchProject(projectId),
});
```

### 3. Via Events (Rare)

```typescript
// Only for cross-cutting concerns like notifications
window.dispatchEvent(
    new CustomEvent('annotation-saved', {
        detail: { annotationId: '123' },
    })
);
```

## Architecture consequences

### Positive

- ✅ **Easy to find code**: Everything for a feature is in one place
- ✅ **Easy to delete**: Remove feature by deleting one folder
- ✅ **Team autonomy**: Teams can own entire features
- ✅ **Reduced coupling**: Features can't accidentally depend on each other
- ✅ **Faster onboarding**: Clear feature boundaries
- ✅ **Parallel development**: Multiple devs work on different features without conflicts

### Negative

- ⚠️ **Duplication risk**: Features might duplicate code instead of sharing
- ⚠️ **Judgment required**: "Is this truly shared?" decisions
- ⚠️ **Refactoring cost**: Moving code between shared/feature can be tedious
- ⚠️ **Import paths**: More `../../` or need path aliases

### Neutral

- Not a silver bullet - still requires discipline
- Works best with domain-driven thinking
- May feel unfamiliar to developers used to MVC

## Best Practices

### 1. Colocation

Place code next to where it's used:

```
✅ features/annotator/hooks/use-draw-rectangle.ts
❌ src/hooks/use-draw-rectangle.ts (if only annotator uses it)
```

### 2. Limit Feature Scope

Features should be **cohesive** but not too broad:

```
✅ features/annotator/          # Focused on annotation
✅ features/dataset/            # Focused on media management

❌ features/app/                # Too broad
❌ features/utils/              # Not a feature
```

### 3. Public API Discipline

Only export what's truly needed:

```typescript
// ✅ Minimal public API
export { AnnotatorProvider, useAnnotationActions };

// ❌ Exposing internals
export { Canvas, Toolbar, DrawRectangle, ... }; // Too much
```

### 4. Use Path Aliases

```json
// tsconfig.json
{
    "compilerOptions": {
        "paths": {
            "@/api/*": ["./src/api/*"],
            "@/features/*": ["./src/features/*"],
            "@/components/*": ["./src/components/*"]
        }
    }
}
```

```typescript
// ✅ Clean imports
import { Button } from '@geti/ui';
import { fetchAnnotations } from '@/api/annotations';
import { useAnnotationActions } from '@/features/annotator';

// ❌ Messy imports
import { Button } from '../../../packages/ui';
import { fetchAnnotations } from '../../../api/annotations';
```

## Migration Strategy

Moving from layered to feature-based:

1. **Identify features** (annotator, dataset, inference, project)
2. **Create feature folders** (`features/annotator/`)
3. **Move components one at a time**
4. **Update imports** as you go
5. **Extract shared code last** (wait until patterns emerge)

## References

- [Feature-Sliced Design](https://feature-sliced.design/)
- [Domain-Driven Design Frontend](https://khalilstemmler.com/articles/client-side-architecture/introduction/)
- [Bulletproof React](https://github.com/alan2207/bulletproof-react)
- [Screaming Architecture](https://blog.cleancoder.com/uncle-bob/2011/09/30/Screaming-Architecture.html)
