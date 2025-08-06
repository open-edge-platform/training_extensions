import { expect, test } from './fixtures';

test.describe('data collection', () => {
    test('list items', async ({ page }) => {
        await page.goto('/data-collection');

        await expect(page.getByRole('button', { name: 'Edit collection criteria' })).toBeVisible();
        await expect(page.getByText('Visualize and manage your')).toBeVisible();
    });
});
