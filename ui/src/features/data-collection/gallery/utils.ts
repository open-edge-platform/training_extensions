import { API_BASE_URL } from '../../../api/client';

const getBaseUrl = (src: string) => `${API_BASE_URL}/api/data-collection/${src}`;

export const getImageUrl = (src: string) => {
    return `${getBaseUrl(src)}/image`;
};

export const getThumbnailUrl = (src: string) => {
    return `${getImageUrl(src)}-thumbnail`;
};

export const getPredictionThumbnailUrl = (src: string) => {
    return `${getBaseUrl(src)}/prediction-thumbnail`;
};
