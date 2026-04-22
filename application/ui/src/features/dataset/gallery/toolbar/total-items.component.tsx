// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Text } from '@geti/ui';
import { useDatasetStatistics } from 'hooks/api/dataset.hook';

type TotalItemsProps = {
    totalSelectedElements: number;
};

export const TotalItems = ({ totalSelectedElements }: TotalItemsProps) => {
    const { data: statistics } = useDatasetStatistics();

    const hasSelectedElements = totalSelectedElements > 0;
    const numberOfImages = statistics?.media_counts.images;
    const numberOfVideos = statistics?.media_counts.videos;
    const imagesMessage = `${numberOfImages} image${numberOfImages === 1 ? '' : 's'}`;
    const videosMessage = `${numberOfVideos} video${numberOfVideos === 1 ? '' : 's'}`;

    if (hasSelectedElements) {
        return <Text>{`${totalSelectedElements} selected`}</Text>;
    }

    if (numberOfImages > 0 && numberOfVideos > 0) {
        return <Text>{`${imagesMessage}, ${videosMessage}`}</Text>;
    }

    if (numberOfVideos > 0) {
        return <Text>{videosMessage}</Text>;
    }
    if (numberOfImages > 0) {
        return <Text>{imagesMessage}</Text>;
    }

    return '';
};
