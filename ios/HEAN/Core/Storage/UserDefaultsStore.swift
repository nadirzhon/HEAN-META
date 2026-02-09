//
//  UserDefaultsStore.swift
//  HEAN
//

import Foundation

final class UserDefaultsStore {
    static let shared = UserDefaultsStore()
    private let defaults = UserDefaults.standard

    private init() {}

    // MARK: - API Configuration

    var apiBaseURL: String {
        get { defaults.string(forKey: "api_base_url") ?? "http://localhost:8000" }
        set { defaults.set(newValue, forKey: "api_base_url") }
    }

    var wsBaseURL: String {
        get { defaults.string(forKey: "ws_base_url") ?? "ws://localhost:8000/ws" }
        set { defaults.set(newValue, forKey: "ws_base_url") }
    }

    var apiAuthKey: String? {
        get { KeychainStore.shared.get("api_auth_key") }
        set {
            if let value = newValue {
                KeychainStore.shared.save(value, for: "api_auth_key")
            } else {
                KeychainStore.shared.delete("api_auth_key")
            }
        }
    }

    // MARK: - UI Preferences

    var showConfidenceScores: Bool {
        get { defaults.bool(forKey: "show_confidence_scores") }
        set { defaults.set(newValue, forKey: "show_confidence_scores") }
    }

    var enableHaptics: Bool {
        get { defaults.object(forKey: "enable_haptics") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "enable_haptics") }
    }

    var enableNotifications: Bool {
        get { defaults.object(forKey: "enable_notifications") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "enable_notifications") }
    }

    // MARK: - First Launch

    var hasCompletedOnboarding: Bool {
        get { defaults.bool(forKey: "has_completed_onboarding") }
        set { defaults.set(newValue, forKey: "has_completed_onboarding") }
    }
}
