// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import thumbnailUrl from '../../assets/mocked-project-thumbnail.png';
import { response } from '../dataset/mock-response';

type Item = (typeof response.items)[number];

export const ImageAnnotations = ({ mediaItem }: { mediaItem: Item }) => {
    return (
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
            <img
                src={thumbnailUrl}
                alt={mediaItem.original_name}
                style={{ objectFit: 'cover', width: '100%', height: '100%' }}
            />
        </div>
    );
};
