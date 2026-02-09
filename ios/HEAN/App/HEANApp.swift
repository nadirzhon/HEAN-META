//
//  HEANApp.swift
//  HEAN
//
//  Premium iOS Trading App
//

import SwiftUI

@main
struct HEANApp: App {
    @StateObject private var container = DIContainer.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(container)
                .preferredColorScheme(.dark)
                .task {
                    container.start()
                }
        }
    }
}

// MARK: - Preview
#Preview {
    ContentView()
        .environmentObject(DIContainer.shared)
        .preferredColorScheme(.dark)
}
