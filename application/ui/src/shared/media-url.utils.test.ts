// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// `API_BASE_URL` is read at module load time, so each scenario uses
// `vi.doMock` + `vi.resetModules` to re-import the module under test with a
// different value. This covers:
//   - Dev/preview: absolute base (e.g. `http://localhost:7860`)
//   - Docker:      empty base (UI served by FastAPI on the same origin)

afterEach(() => {
    vi.resetModules();
    vi.doUnmock('../api/client');
});

const loadHelpers = async (apiBaseUrl: string) => {
    vi.doMock('../api/client', () => ({ API_BASE_URL: apiBaseUrl }));
    return import('./media-url.utils');
};

describe.each([
    { name: 'absolute base (dev/preview)', base: 'http://localhost:7860', prefix: 'http://localhost:7860' },
    { name: 'absolute base with trailing slash', base: 'http://localhost:7860/', prefix: 'http://localhost:7860' },
    { name: 'empty base (Docker, same origin)', base: '', prefix: '' },
])('media-url helpers with $name', ({ base, prefix }) => {
    it('does not throw when constructing URLs', async () => {
        await expect(loadHelpers(base)).resolves.toBeDefined();
    });

    it('builds the expected URLs', async () => {
        const helpers = await loadHelpers(base);

        expect(helpers.getProjectThumbnailUrl('p1')).toBe(`${prefix}/api/projects/p1/thumbnail`);
        expect(helpers.getThumbnailUrl('p1', 'm1')).toBe(`${prefix}/api/projects/p1/dataset/media/m1/thumbnail`);
        expect(helpers.getMediaBinaryUrl('p1', 'm1')).toBe(`${prefix}/api/projects/p1/dataset/media/m1/binary`);
        expect(helpers.getDatasetRevisionThumbnailUrl('p1', 'r1', 'm1')).toBe(
            `${prefix}/api/projects/p1/dataset_revisions/r1/items/m1/thumbnail`
        );
        expect(helpers.getVideoFrameBinaryUrl('p1', 'm1', 7)).toBe(
            `${prefix}/api/projects/p1/dataset/media/m1/binary?frame_index=7`
        );
        expect(helpers.getVideoFrameThumbnailUrl('p1', 'm1', 7)).toBe(
            `${prefix}/api/projects/p1/dataset/media/m1/thumbnail?frame_index=7`
        );
    });
});
