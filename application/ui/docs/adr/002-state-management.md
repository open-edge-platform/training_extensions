# ADR 002: State Management with Context Providers

## Context

We needed a state management solution that:

- Handles complex domain logic (annotations, zoom, selections)
- Avoids prop drilling
- Works well with React Query for server state
- Keeps components decoupled
- Supports multiple independent state domains

## Decision

Use **React Context API with custom providers** for domain-specific state, and **React Query** for server state.

### Architecture

```
State Layer:
├── Server State (React Query)
│   └── API data, caching, mutations
└── Client State (Context Providers)
    ├── AnnotationActionsProvider (CRUD operations)
    ├── AnnotationVisibilityProvider (UI state)
    ├── SelectAnnotationProvider (selection state)
    ├── AnnotatorProvider (tool state)
    └── ZoomProvider (zoom/pan state)
```

## Implementation

### Provider Pattern

```typescript
// annotation-actions-provider.component.tsx
interface AnnotationsContextValue {
    annotations: Annotation[];
    addAnnotations: (shapes: Shape[]) => void;
    deleteAnnotations: (ids: string[]) => void;
    updateAnnotations: (annotations: Annotation[]) => void;
}

const AnnotationsContext = createContext<AnnotationsContextValue | null>(null);

export const AnnotationActionsProvider = ({ children, mediaItem }) => {
    const [localAnnotations, setLocalAnnotations] = useState<Annotation[]>([]);

    const addAnnotations = (shapes: Shape[]) => {
        setLocalAnnotations(prev => [...prev, ...shapes.map(toAnnotation)]);
    };

    return (
        <AnnotationsContext.Provider value={{ annotations: localAnnotations, addAnnotations, ... }}>
            {children}
        </AnnotationsContext.Provider>
    );
};

export const useAnnotationActions = () => {
    const context = useContext(AnnotationsContext);
    if (!context) throw new Error('Must use within provider');
    return context;
};
```

### Provider Composition

```tsx
// media-preview.component.tsx
<AnnotationActionsProvider mediaItem={mediaItem}>
    <ZoomProvider>
        <SelectAnnotationProvider>
            <AnnotationVisibilityProvider>
                <AnnotatorProvider>{/* Components access any provider via hooks */}</AnnotatorProvider>
            </AnnotationVisibilityProvider>
        </SelectAnnotationProvider>
    </ZoomProvider>
</AnnotationActionsProvider>
```

## Provider Responsibilities

### AnnotationActionsProvider

**Purpose**: Manages annotation data and CRUD operations  
**State**: `localAnnotations[]`, `isDirty`  
**Actions**: `addAnnotations()`, `updateAnnotations()`, `deleteAnnotations()`, `submitAnnotations()`  
**Syncs with**: Server via React Query

### AnnotationVisibilityProvider

**Purpose**: UI visibility state  
**State**: `isVisible`, `isFocussed`  
**Actions**: `toggleVisibility()`, `toggleFocus()`  
**Pure UI**: No server interaction

### SelectAnnotationProvider

**Purpose**: Selection state  
**State**: `selectedAnnotations: Set<string>`  
**Actions**: `setSelectedAnnotations()`, `toggleSelection()`  
**Supports**: Multi-select, click handlers

### ZoomProvider

**Purpose**: Canvas zoom/pan transformations  
**State**: `scale`, `translate`, `canvasSize`  
**Actions**: `setZoom()`, `fitToScreen()`, `zoomIn()`, `zoomOut()`  
**Math**: Transform calculations

## Alternatives Considered

### 1. Zustand

- ❌ Another dependency
- ❌ Less explicit than Context
- ✅ Simpler API than Redux
- ✅ Good performance

### 2. Props Drilling

- ❌ Unmaintainable for deep hierarchies
- ❌ Violates component boundaries
- ✅ Simple, explicit
- ✅ No magic

## Consequences

### Positive

- ✅ **Scoped state**: Each provider owns its domain
- ✅ **No prop drilling**: Components access state directly
- ✅ **Testable**: Providers can be tested in isolation
- ✅ **Composable**: Mix and match providers as needed
- ✅ **Type-safe**: Full TypeScript support
- ✅ **React-native**: Uses standard React patterns

### Negative

- ⚠️ **Provider hell**: Deep nesting can be verbose
- ⚠️ **Re-render optimization**: Need `useMemo`/`useCallback` carefully
- ⚠️ **No DevTools**: Unlike Redux, no time-travel debugging
- ⚠️ **Testing setup**: Each test needs provider wrapper

### Neutral

- Context API is built-in (no extra dependencies)
- Performance is good enough for our use case
- Can migrate to Zustand/Redux later if needed

## Best Practices

### 1. Single Responsibility

Each provider manages ONE domain:

```typescript
// ✅ Good
AnnotationVisibilityProvider; // Only visibility state

// ❌ Bad
AnnotationProvider; // Too broad, mixes concerns
```

### 2. Custom Hooks

Always provide a custom hook:

```typescript
export const useAnnotationActions = () => {
    const context = useContext(AnnotationsContext);
    if (!context) throw new Error('Provider missing');
    return context;
};
```

### 3. Minimize Re-renders

```typescript
const value = useMemo(
    () => ({
        annotations,
        addAnnotations,
        deleteAnnotations,
    }),
    [annotations]
); // Only recreate if annotations change
```

### 4. Provider Placement

Place providers as close as possible to consumers:

```typescript
// ✅ Scoped to feature
<MediaPreview>
    <AnnotationActionsProvider>
        <AnnotatorCanvas />
    </AnnotationActionsProvider>
</MediaPreview>

// ❌ Too global
<App>
    <AnnotationActionsProvider> {/* Unnecessary for non-annotation pages */}
        <Router />
    </AnnotationActionsProvider>
</App>
```

## Migration Path

If we outgrow Context, migration path:

1. Keep provider interfaces unchanged
2. Swap Context implementation with Zustand/Redux
3. Consumers continue using custom hooks
4. No component changes needed

## References

- [React Context API](https://react.dev/reference/react/createContext)
- [Context Performance](https://react.dev/reference/react/useMemo#skipping-expensive-recalculations)
- [Composition Pattern](https://react.dev/learn/passing-data-deeply-with-context)
