// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest';

import { isValidIp } from './utils';

describe('IpCamera utils', () => {
    it('returns true for valid IPv4 and IPv6 addresses using isValidIp', () => {
        // Valid IPv4
        expect(isValidIp('192.168.1.1')).toBe(true);
        expect(isValidIp('0.0.0.0')).toBe(true);
        expect(isValidIp('255.255.255.255')).toBe(true);
        // Valid IPv6
        expect(isValidIp('2001:0db8:85a3:0000:0000:8a2e:0370:7334')).toBe(true);
        expect(isValidIp('fe80:0000:0000:0000:0202:b3ff:fe1e:8329')).toBe(true);
        expect(isValidIp('::1:2:3:4:5:6:7')).toBe(false); // Not matching strict regex
    });

    it('returns false for invalid IP addresses using isValidIp', () => {
        expect(isValidIp('256.256.256.256')).toBe(false);
        expect(isValidIp('192.168.1')).toBe(false);
        expect(isValidIp('192.168.1.1.1')).toBe(false);
        expect(isValidIp('abc.def.ghi.jkl')).toBe(false);
        expect(isValidIp('')).toBe(false);
        expect(isValidIp('2001:db8:85a3::8a2e:370:7334')).toBe(false); // compressed IPv6 not supported by regex
        expect(isValidIp('1234:5678:9abc:def0:1234:5678:9abc:defg')).toBe(false); // invalid hex
    });
});
