// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

export const isValidIp4 = (ip: string): boolean => {
    const ipv4 = /^(25[0-5]|2[0-4]\d|[01]?\d\d?)(\.(25[0-5]|2[0-4]\d|[01]?\d\d?)){3}$/;
    return ipv4.test(ip);
};

export const isValidIp6 = (ip: string): boolean => {
    const ipv6 = /^(([0-9a-fA-F]{1,4}):){7}([0-9a-fA-F]{1,4})$/;
    return ipv6.test(ip);
};

export const isValidIp = (ip: string) => {
    return isValidIp6(ip) || isValidIp4(ip);
};
