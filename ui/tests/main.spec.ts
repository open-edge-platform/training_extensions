import { expect, test } from './fixtures';

test.describe('livefeed', () => {
    test('starts stream', async ({ page }) => {
        await page.goto('/live-feed');

        await expect(page.getByLabel('Idle')).toBeVisible();

        await page.getByLabel('Start stream').click();

        // TODO: fix the stream mock and update this to "Connected"
        await expect(page.getByLabel('Connecting')).toBeVisible();
    });
});
