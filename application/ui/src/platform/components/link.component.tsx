// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { type ComponentProps } from 'react';

import { Link as GetiLink } from '@geti-ui/ui';

type LinkProps = ComponentProps<typeof GetiLink>;

export const Link = (props: LinkProps) => <GetiLink {...props} />;
