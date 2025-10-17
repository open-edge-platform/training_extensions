# ADR 003: Testing Strategy

## Context

We needed a comprehensive testing strategy that:

- Catches bugs early in development
- Validates user workflows end-to-end
- Runs fast in CI/CD pipeline
- Provides confidence for refactoring
- Balances coverage with maintainability

## Decision

Use a **three-tier testing pyramid**:

1. **Unit Tests** (Vitest) - Component logic, utilities, hooks
2. **Component Tests** (Playwright Component) - Visual components, interactions
3. **E2E Tests** (Playwright) - User workflows, critical paths

### Testing Pyramid

```
         /\
        /  \  E2E (Playwright)
       /    \  - User workflows
      /------\  - Critical paths
     /        \ - ~10-20 tests
    /----------\
   / Component \ (Playwright Component)
  /   Tests    \ - UI interactions
 /--------------\ - ~30-50 tests
/                \
/   Unit Tests   \ (Vitest)
/    (Vitest)    \ - Logic, utils, hooks
/------------------\ - ~100+ tests
```

## Implementation

### 1. Unit Tests (Vitest)

**Purpose**: Test business logic, utilities, custom hooks in isolation

```typescript
// src/hooks/use-zoom.test.ts

import { act, renderHook } from '@testing-library/react';

import { useZoom } from './use-zoom';

describe('useZoom', () => {
    it('should initialize with default scale', () => {
        const { result } = renderHook(() => useZoom());
        expect(result.current.scale).toBe(1);
    });

    it('should zoom in by 10%', () => {
        const { result } = renderHook(() => useZoom());
        act(() => result.current.zoomIn());
        expect(result.current.scale).toBe(1.1);
    });

    it('should not zoom beyond max scale', () => {
        const { result } = renderHook(() => useZoom({ maxScale: 5 }));
        act(() => {
            for (let i = 0; i < 100; i++) result.current.zoomIn();
        });
        expect(result.current.scale).toBe(5);
    });
});
```

**Run**: `npm test`

### 2. Component Tests (Playwright Component)

**Purpose**: Test component rendering, user interactions, visual states

```typescript
// src/components/annotation-labels.test.tsx
import { test, expect } from '@playwright/experimental-ct-react';
import { AnnotationLabels } from './annotation-labels.component';

test('should render label with correct text', async ({ mount }) => {
    const component = await mount(
        <AnnotationLabels
            labels={[{ id: '1', name: 'Person', color: '#FF0000' }]}
            position={{ x: 100, y: 50 }}
            scale={1}
        />
    );

    await expect(component.getByText('Person')).toBeVisible();
});
```

**Run**: `npm run test:component`

### 3. E2E Tests (Playwright)

**Purpose**: Test complete user workflows across the application

```typescript
// tests/e2e/annotation-workflow.spec.ts

import { expect, test } from '@playwright/test';

test.describe('Annotation Workflow', () => {
    test('should create, edit, and delete annotation', async ({ page }) => {
        await page.goto('/projects/123/annotate');

        // Create annotation
        await page.getByRole('button', { name: 'Rectangle' }).click();
        await page.locator('canvas').click({ position: { x: 100, y: 100 } });
        await page.locator('canvas').click({ position: { x: 300, y: 300 } });

        // Add label
        await page.getByRole('combobox', { name: /label/i }).fill('Person');
        await page.keyboard.press('Enter');

        // Verify annotation exists
        await expect(page.getByText('Person')).toBeVisible();

        // Submit annotations
        await page.getByRole('button', { name: 'Save' }).click();
        await expect(page.getByText('Saved successfully')).toBeVisible();

        // Delete annotation
        await page.getByText('Person').click();
        await page.keyboard.press('Delete');
        await expect(page.getByText('Person')).not.toBeVisible();
    });
});
```

**Run**: `npm run test:e2e`

## Testing Configuration

### vitest.config.ts

```typescript
export default defineConfig({
    test: {
        environment: 'jsdom',
        setupFiles: ['./src/setup-tests.ts'],
        coverage: {
            provider: 'v8',
            reporter: ['text', 'json', 'html'],
            exclude: ['**/*.test.{ts,tsx}', '**/mocks/**', '**/*.d.ts'],
        },
    },
});
```

### playwright.config.ts

```typescript
export default defineConfig({
    testDir: './tests',
    use: {
        baseURL: 'http://localhost:3000',
        screenshot: 'only-on-failure',
        video: 'retain-on-failure',
    },
    webServer: {
        command: 'npm run dev',
        port: 3000,
        reuseExistingServer: !process.env.CI,
    },
});
```

## Alternatives Considered

### 1. Jest

- ❌ Slower than Vitest
- ❌ More configuration required
- ✅ Industry standard, more resources
- ✅ Mature ecosystem

### 2. Cypress

- ❌ Slower test execution than Playwright
- ❌ Less powerful component testing
- ✅ Great debugging UI
- ✅ Time-travel debugging

## Consequences

### Positive

- ✅ **Fast feedback**: Vitest runs in <2s for unit tests
- ✅ **Confidence**: E2E tests catch integration bugs
- ✅ **DX**: Playwright UI mode for debugging
- ✅ **Type-safe**: Full TypeScript support
- ✅ **CI-friendly**: Parallel execution, retries
- ✅ **Real browser**: Playwright uses actual Chromium/Firefox

### Negative

- ⚠️ **Maintenance**: E2E tests can be brittle
- ⚠️ **Slow E2E**: Full workflows take 30s-2min
- ⚠️ **Learning curve**: Two test frameworks to learn
- ⚠️ **CI cost**: E2E tests require more resources

## Testing Best Practices

### 1. Test User Behavior, Not Implementation

```typescript
// ✅ Good - Tests behavior
test('should add label when Enter is pressed', async ({ page }) => {
    await page.getByRole('textbox').fill('Person');
    await page.keyboard.press('Enter');
    await expect(page.getByText('Person')).toBeVisible();
});

// ❌ Bad - Tests implementation
test('should call addLabel function', () => {
    const addLabel = vi.fn();
    render(<LabelInput onAddLabel={addLabel} />);
    // ... test calls addLabel
});
```

### 2. Use Data-Testid Sparingly

```typescript
// ✅ Prefer semantic selectors
await page.getByRole('button', { name: 'Save' });
await page.getByLabel('Label name');

// ⚠️ Use data-testid only when necessary
await page.getByTestId('annotation-canvas');
```

### 3. Arrange-Act-Assert Pattern

```typescript
test('should zoom in when button clicked', () => {
    // Arrange
    const { result } = renderHook(() => useZoom());

    // Act
    act(() => result.current.zoomIn());

    // Assert
    expect(result.current.scale).toBe(1.1);
});
```

### 4. Mock External Dependencies

```typescript
// setup-tests.ts
vi.mock('./api/client', () => ({
    fetchAnnotations: vi.fn(() => Promise.resolve([])),
}));
```

### 5. Test Error States

```typescript
test('should show error toast on upload failure', async ({ page }) => {
    await page.route('**/api/media', (route) => route.abort());
    await page.getByRole('button', { name: 'Upload' }).click();
    await expect(page.getByText(/upload failed/i)).toBeVisible();
});
```

## Coverage Targets

- **Unit Tests**: >80% coverage for utilities, hooks
- **Component Tests**: Critical user-facing components
- **E2E Tests**: All primary user workflows

### Running Coverage

```bash
npm run test:coverage
```

## CI Integration

```yaml
# .github/workflows/test.yml
- name: Run unit tests
  run: npm test -- --coverage

- name: Run component tests
  run: npm run test:component

- name: Run E2E tests
  run: npm run test:e2e
```

## References

- [Vitest Documentation](https://vitest.dev/)
- [Playwright Documentation](https://playwright.dev/)
- [Testing Library](https://testing-library.com/)
- [Playwright Component Testing](https://playwright.dev/docs/test-components)
