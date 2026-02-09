//
//  APIError.swift
//  HEAN
//

import Foundation

enum APIError: LocalizedError {
    case invalidURL
    case invalidResponse
    case httpError(statusCode: Int, message: String?)
    case decodingError(Error)
    case networkError(Error)
    case unauthorized
    case rateLimited
    case serverError

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid URL"
        case .invalidResponse: return "Invalid server response"
        case .httpError(let code, let msg): return "HTTP \(code): \(msg ?? "Unknown error")"
        case .decodingError(let err): return "Data parsing error: \(err.localizedDescription)"
        case .networkError(let err): return "Network error: \(err.localizedDescription)"
        case .unauthorized: return "Authentication required"
        case .rateLimited: return "Too many requests. Please wait."
        case .serverError: return "Server error. Please try again."
        }
    }

    var isRetryable: Bool {
        switch self {
        case .rateLimited, .serverError, .networkError: return true
        default: return false
        }
    }
}
