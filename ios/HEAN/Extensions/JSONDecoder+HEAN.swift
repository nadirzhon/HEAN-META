//
//  JSONDecoder+HEAN.swift
//  HEAN
//

import Foundation

extension JSONDecoder {
    /// Pre-configured decoder for HEAN API responses
    static let hean: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let dateString = try container.decode(String.self)

            // Try ISO 8601 first
            if let date = ISO8601DateFormatter().date(from: dateString) {
                return date
            }

            // Try common formats (order: most specific to least)
            let formats = [
                "yyyy-MM-dd'T'HH:mm:ss.SSSSSSZZZZZ",  // microseconds + timezone
                "yyyy-MM-dd'T'HH:mm:ss.SSSZZZZZ",     // milliseconds + timezone
                "yyyy-MM-dd'T'HH:mm:ssZZZZZ",          // no fractional + timezone
                "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",        // microseconds, no timezone (Python default)
                "yyyy-MM-dd'T'HH:mm:ss.SSS",           // milliseconds, no timezone
                "yyyy-MM-dd'T'HH:mm:ss",               // no fractional, no timezone
                "yyyy-MM-dd HH:mm:ss"                   // space separator
            ]

            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")

            for format in formats {
                formatter.dateFormat = format
                if let date = formatter.date(from: dateString) {
                    return date
                }
            }

            // Try Unix timestamp
            if let timestamp = Double(dateString) {
                return Date(timeIntervalSince1970: timestamp)
            }

            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: "Unable to decode date: \(dateString)"
            ))
        }
        return decoder
    }()
}

extension JSONEncoder {
    static let hean: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()
}
