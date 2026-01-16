// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from 'hooks/use-project-identifier.hook';

import { $api } from '../../../../api/client';

const mockedModels: Model[] = [
    {
        id: '76e07d18-196e-4e33-bf98-ac1d35dca4cb',
        name: 'Object_Detection_YOLOX_X (76e07d18)',
        architecture: 'Object_Detection_YOLOX_X',
        parent_revision: null,
        training_info: {
            status: 'successful',
            label_schema_revision: {
                labels: [
                    { id: 'a22d82ba-afa9-4d6e-bbc1-8c8e4002ec29', name: 'cat' },
                    { id: '8aa85368-11ba-4507-88f2-6a6704d78ef5', name: 'dog' },
                ],
            },
            configuration: {},
            start_time: '2025-01-10T10:00:00.000000+00:00',
            end_time: '2025-01-10T12:30:00.000000+00:00',
            dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
        },
        files_deleted: false,
    },
    {
        id: '76e07d18-196e-4e33-bf98-ac1d35dca4cc',
        name: 'Object_Detection_YOLOX_Y (76e07d18)',
        architecture: 'Object_Detection_YOLOX_Y',
        parent_revision: '76e07d18-196e-4e33-bf98-ac1d35dca4cb',
        training_info: {
            status: 'successful',
            label_schema_revision: {
                labels: [
                    { id: 'a22d82ba-afa9-4d6e-bbc1-8c8e4002ec29', name: 'cat' },
                    { id: '8aa85368-11ba-4507-88f2-6a6704d78ef5', name: 'dog' },
                ],
            },
            configuration: {},
            start_time: '2025-01-10T10:00:00.000000+00:00',
            end_time: '2025-01-10T12:30:00.000000+00:00',
            dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
        },
        files_deleted: false,
    },
    {
        id: 'a1b2c3d4-5678-90ab-cdef-1234567890ab',
        name: 'Image_Classification_ResNet50 (a1b2c3d4)',
        architecture: 'Image_Classification_ResNet50',
        parent_revision: null,
        training_info: {
            status: 'in_progress',
            label_schema_revision: {
                labels: [
                    { id: 'b33d92ca-bfa9-5d7e-ccd2-9d9e5003ed30', name: 'bird' },
                    { id: 'c44e03db-c0ba-6e8f-dde3-0e0f6114fe41', name: 'fish' },
                    { id: 'd55f14ec-d1cb-7f90-eef4-1f107225gf52', name: 'rabbit' },
                ],
            },
            configuration: {},
            start_time: '2025-01-15T08:30:00.000000+00:00',
            end_time: null,
            dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
        },
        files_deleted: false,
    },
    {
        id: 'f5e4d3c2-b1a0-9876-5432-fedcba098765',
        name: 'Semantic_Segmentation_UNet (f5e4d3c2)',
        architecture: 'Semantic_Segmentation_UNet',
        parent_revision: '76e07d18-196e-4e33-bf98-ac1d35dca4cb',
        training_info: {
            status: 'failed',
            label_schema_revision: {
                labels: [
                    { id: 'e66g25fd-e2dc-8ga1-ffg5-2g218336hg63', name: 'background' },
                    { id: 'f77h36ge-f3ed-9hb2-ggh6-3h329447ih74', name: 'foreground' },
                ],
            },
            configuration: {},
            start_time: '2025-01-12T14:00:00.000000+00:00',
            end_time: '2025-01-12T14:45:00.000000+00:00',
            dataset_revision_id: '3c6c6d38-1cd8-4458-b759-b9880c048b78',
        },
        files_deleted: true,
    },
    {
        id: '12345678-abcd-ef01-2345-6789abcdef01',
        name: 'Instance_Segmentation_MaskRCNN (12345678)',
        architecture: 'Instance_Segmentation_MaskRCNN',
        parent_revision: null,
        training_info: {
            status: 'successful',
            label_schema_revision: {
                labels: [
                    { id: 'g88i47hf-g4fe-0ic3-hhi7-4i430558ji85', name: 'person' },
                    { id: 'h99j58ig-h5gf-1jd4-iij8-5j541669kj96', name: 'car' },
                    { id: 'i00k69jh-i6hg-2ke5-jjk9-6k652770lk07', name: 'bicycle' },
                    { id: 'j11l70ki-j7ih-3lf6-kkl0-7l763881ml18', name: 'motorcycle' },
                ],
            },
            configuration: {},
            start_time: '2025-01-08T09:15:00.000000+00:00',
            end_time: '2025-01-08T16:45:00.000000+00:00',
            dataset_revision_id: '6f9f9g61-4fg1-7781-e082-e1113f371e01',
        },
        files_deleted: false,
    },
];

export const useGetModels = () => {
    const projectId = useProjectIdentifier();

    // return $api.useSuspenseQuery('get', '/api/projects/{project_id}/models', {
    //     params: { path: { project_id: projectId } },
    // });

    return {
        data: mockedModels,
    };
};
