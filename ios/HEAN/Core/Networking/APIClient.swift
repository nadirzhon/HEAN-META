//
//  APIClient.swift
//  HEAN
//
//  Created by HEAN Team on 2026-01-31.
//

import Foundation
import OSLog

enum APIError: Error, LocalizedError {
    case invalidURL
    case networkError(Error)
    case invalidResponse
    case httpError(statusCode: Int, message: String?)
    case decodingError(Error)
    case maxRetriesExceeded

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .invalidResponse:
            return "Invalid response from server"
        case .httpError(let code, let message):
            return "HTTP \(code): \(message ?? "Unknown error")"
        case .decodingError(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .maxRetriesExceeded:
            return "Maximum retry attempts exceeded"
        }
    }
}

actor APIClient {
    private let baseURL: String
    private let session: URLSession
    private let maxRetries = 1
    private let retryDelay: TimeInterval = 0.5

    init(baseURL: String) {
        self.baseURL = baseURL

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 5
        config.timeoutIntervalForResource = 10
        config.requestCachePolicy = .reloadIgnoringLocalCacheData

        self.session = URLSession(configuration: config)
    }

    func request<T: Decodable>(
        _ endpoint: APIEndpoint,
        retryCount: Int = 0
    ) async throws -> T {
        guard let url = URL(string: baseURL + endpoint.path) else {
            throw APIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = endpoint.method.rawValue
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let body = endpoint.body {
            request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        }

        do {
            let (data, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.invalidResponse
            }

            Logger.network.debug("[\(endpoint.method.rawValue)] \(endpoint.path) -> \(httpResponse.statusCode)")

            guard (200...299).contains(httpResponse.statusCode) else {
                let message = String(data: data, encoding: .utf8)
                throw APIError.httpError(statusCode: httpResponse.statusCode, message: message)
            }

            do {
                return try JSONDecoder.hean.decode(T.self, from: data)
            } catch {
                throw APIError.decodingError(error)
            }

        } catch {
            // Retry logic for network errors
            if retryCount < self.maxRetries && shouldRetry(error) {
                Logger.network.warning("Request failed, retrying (\(retryCount + 1)/\(self.maxRetries))...")
                try await Task.sleep(for: .seconds(retryDelay * Double(retryCount + 1)))
                return try await self.request(endpoint, retryCount: retryCount + 1)
            }

            if retryCount >= self.maxRetries {
                throw APIError.maxRetriesExceeded
            }

            throw APIError.networkError(error)
        }
    }

    private func shouldRetry(_ error: Error) -> Bool {
        // Retry on network errors, but not on client errors (4xx)
        if let apiError = error as? APIError {
            if case .httpError(let code, _) = apiError {
                return code >= 500 // Retry only on server errors
            }
        }
        return true
    }
}

enum HTTPMethod: String {
    case GET, POST, PUT, DELETE, PATCH
}

protocol APIEndpoint {
    var path: String { get }
    var method: HTTPMethod { get }
    var body: [String: Any]? { get }
}

// MARK: - Convenience Methods

extension APIClient {
    /// GET request convenience method
    func get<T: Decodable>(_ path: String) async throws -> T {
        try await request(SimpleEndpoint(path: path, method: .GET, body: nil))
    }

    /// POST request convenience method
    func post<T: Decodable>(_ path: String, body: [String: Any]) async throws -> T {
        try await request(SimpleEndpoint(path: path, method: .POST, body: body))
    }

    /// PUT request convenience method
    func put<T: Decodable>(_ path: String, body: [String: Any]) async throws -> T {
        try await request(SimpleEndpoint(path: path, method: .PUT, body: body))
    }

    /// DELETE request convenience method
    func delete<T: Decodable>(_ path: String) async throws -> T {
        try await request(SimpleEndpoint(path: path, method: .DELETE, body: nil))
    }
}

/// Simple endpoint for convenience methods
private struct SimpleEndpoint: APIEndpoint {
    let path: String
    let method: HTTPMethod
    let body: [String: Any]?
}
