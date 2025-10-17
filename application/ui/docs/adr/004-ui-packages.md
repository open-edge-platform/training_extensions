# ADR 004: UI Packages Architecture

## Context

We needed a UI component system that:

- Provides accessible, production-ready components
- Maintains consistency across the application
- Supports theming and customization
- Avoids reinventing common patterns
- Integrates with our React/TypeScript stack

Additionally, we wanted to share this system with related projects (Geti) and organize shared utilities.

## Decision

Use **@geti/ui** (Adobe React Spectrum based) as our primary component library, organized as a monorepo with separate packages:

```
ui/
├── packages/
│   ├── ui/           # @geti/ui - Adobe Spectrum components
│   ├── smart-tools/  # @geti/smart-tools - AI annotation utilities
│   └── config/       # Shared build/lint configuration
```

### Package Responsibilities

#### @geti/ui

- **Purpose**: Core UI component library
- **Based on**: Adobe React Spectrum
- **Contains**: Buttons, Forms, Tables, Dialogs, Navigation, Layout
- **Accessibility**: ARIA compliant out-of-the-box
- **Theming**: CSS variables for customization

#### @geti/smart-tools

- **Purpose**: AI-powered annotation utilities
- **Contains**: Auto-segmentation, object detection helpers
- **Independent**: Can be used without @geti/ui

#### packages/config

- **Purpose**: Shared build configuration
- **Contains**: ESLint, TypeScript, Rsbuild configs
- **DRY**: Single source of truth for tooling

## Implementation

### Package Structure

```
packages/ui/
├── package.json
├── src/
│   ├── index.ts                    # Public API
│   ├── components/
│   │   ├── button/
│   │   │   ├── button.component.tsx
│   │   │   ├── button.test.tsx
│   │   │   └── index.ts
│   │   ├── form/
│   │   ├── table/
│   │   └── ...
│   ├── hooks/
│   │   ├── use-theme.ts
│   │   └── use-media-query.ts
│   └── styles/
│       ├── theme.scss
│       └── variables.scss
```

### Usage in Application

```typescript
// src/features/annotator/toolbar.component.tsx
import {
    Button,
    ActionButton,
    Form,
    TextField,
    TooltipTrigger,
    Tooltip,
} from '@geti/ui';

export const Toolbar = () => {
    return (
        <div className="toolbar">
            <TooltipTrigger>
                <ActionButton onPress={handleRectangle}>
                    <RectangleIcon />
                </ActionButton>
                <Tooltip>Draw Rectangle</Tooltip>
            </TooltipTrigger>

            <Form onSubmit={handleSubmit}>
                <TextField
                    label="Label name"
                    name="labelName"
                    autoComplete="off"
                />
                <Button type="submit">Add Label</Button>
            </Form>
        </div>
    );
};
```

### Workspace Configuration

```json
// package.json
{
    "workspaces": ["packages/*", "src"],
    "dependencies": {
        "@geti/ui": "workspace:*",
        "@geti/smart-tools": "workspace:*"
    }
}
```

### Positive

- ✅ **Accessibility**: WCAG 2.1 AA compliant out of the box
- ✅ **Productivity**: Focus on features, not reinventing buttons
- ✅ **Consistency**: Same components across app
- ✅ **Type-safe**: Full TypeScript support
- ✅ **Maintained**: Adobe maintains the library
- ✅ **Cross-project**: Share with Geti and other tools

### Negative

- ⚠️ **Vendor lock-in**: Hard to migrate away from Spectrum
- ⚠️ **Bundle size**: ~150KB gzipped (reasonable but not tiny)
- ⚠️ **Customization limits**: Some components hard to style deeply
- ⚠️ **Learning curve**: Different API than HTML elements

### Neutral

- Monorepo adds complexity (but provides organization)
- Adobe design language may not match brand
- Community is smaller than MUI/Chakra

## Customization

### Theme Variables

```scss
// packages/ui/src/styles/theme.scss
:root {
    // Colors
    --primary-color: #0078d4;
    --secondary-color: #6c757d;
    --success-color: #28a745;
    --danger-color: #dc3545;

    // Spacing
    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 16px;
    --spacing-lg: 24px;

    // Typography
    --font-family: 'Inter', sans-serif;
    --font-size-sm: 12px;
    --font-size-md: 14px;
    --font-size-lg: 16px;
}
```

### Custom Components

When Spectrum doesn't provide what we need:

```typescript
// packages/ui/src/components/annotation-label/
import { useButton } from '@react-aria/button';
import { useObjectRef } from '@react-aria/utils';

export const AnnotationLabel = ({ label, onDelete }) => {
    // Use React Aria hooks for accessibility
    const ref = useObjectRef(null);
    const { buttonProps } = useButton({ onPress: onDelete }, ref);

    return (
        <div className="annotation-label">
            <span>{label.name}</span>
            <button {...buttonProps} ref={ref}>×</button>
        </div>
    );
};
```

## References

- [Adobe React Spectrum](https://react-spectrum.adobe.com/)
- [React Aria](https://react-spectrum.adobe.com/react-aria/) (Headless version)
- [Spectrum Design System](https://spectrum.adobe.com/)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [npm Workspaces](https://docs.npmjs.com/cli/v7/using-npm/workspaces)
