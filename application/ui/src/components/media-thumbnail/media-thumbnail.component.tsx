// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import styles from './media-thumbnail.module.scss';

type MediaThumbnailProps = {
    onClick?: () => void;
    onDoubleClick?: () => void;
    url: string;
    alt: string;
};

export const MediaThumbnail = ({ onDoubleClick, onClick, url, alt }: MediaThumbnailProps) => {
    return (
        <div onDoubleClick={onDoubleClick} onClick={onClick} className={styles.imgContainer}>
            <img src={url} alt={alt} className={styles.img} />
        </div>
    );
};
