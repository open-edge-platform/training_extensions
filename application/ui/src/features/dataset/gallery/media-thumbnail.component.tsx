// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

type MediaThumbnailProps = {
    onClick?: () => void;
    onDoubleClick?: () => void;
    url: string;
    alt: string;
};

export const MediaThumbnail = ({ onDoubleClick, onClick, url, alt }: MediaThumbnailProps) => {
    return (
        <div onDoubleClick={onDoubleClick} onClick={onClick}>
            <img src={url} alt={alt} style={{ objectFit: 'cover', width: '100%', height: '100%' }} />
        </div>
    );
};
