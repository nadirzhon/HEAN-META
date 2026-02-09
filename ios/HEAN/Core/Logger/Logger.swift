//
//  Logger.swift
//  HEAN
//
//  Created by HEAN Team on 2026-01-31.
//

import Foundation
import OSLog

extension Logger {
    private static let subsystem = Bundle.main.bundleIdentifier ?? "com.hean.trading"

    static let app = Logger(subsystem: subsystem, category: "app")
    static let network = Logger(subsystem: subsystem, category: "network")
    static let websocket = Logger(subsystem: subsystem, category: "websocket")
    static let trading = Logger(subsystem: subsystem, category: "trading")
    static let portfolio = Logger(subsystem: subsystem, category: "portfolio")
    static let ui = Logger(subsystem: subsystem, category: "ui")
}
