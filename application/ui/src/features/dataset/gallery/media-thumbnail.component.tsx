// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

type MediaThumbnailProps = {
    onDoubleClick: () => void;
    url: string;
    alt: string;
};

export const MediaThumbnail = ({ onDoubleClick, url, alt }: MediaThumbnailProps) => {
    return (
        <div onDoubleClick={onDoubleClick}>
            <img src={url} alt={alt} style={{ objectFit: 'cover', width: '100%', height: '100%' }} />
        </div>
    );
};
