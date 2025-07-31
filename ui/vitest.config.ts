import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        environment: 'jsdom',
        // This is needed to use globals like describe or expect
        globals: true,
        include: ['./src/**/*.test.{ts,tsx}'],
        setupFiles: './src/setup-tests.ts',
        watch: false,
        // More context:
        // https://vitest.dev/config/#deps-web-transformcss
        // https://stackoverflow.com/questions/78989267/vitest-unknown-file-extension-css
        // This is needed to set the css transform to 'true' by default
        pool: 'typescript',
    },
});
