//
//  ContentView.swift
//  HEAN
//
//  Main tab navigation
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var container: DIContainer
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            LiveView()
                .tabItem {
                    Label(L.live, systemImage: "bolt.fill")
                }
                .tag(0)

            MindView()
                .tabItem {
                    Label(L.mind, systemImage: "brain.head.profile")
                }
                .tag(1)

            ActionView()
                .tabItem {
                    Label(L.action, systemImage: "target")
                }
                .tag(2)

            XRayView()
                .tabItem {
                    Label(L.xray, systemImage: "eye.fill")
                }
                .tag(3)

            SettingsView()
                .tabItem {
                    Label(L.settings, systemImage: "gearshape.fill")
                }
                .tag(4)
        }
        .tint(Theme.Colors.accent)
    }
}

#Preview {
    ContentView()
        .environmentObject(DIContainer.shared)
        .preferredColorScheme(.dark)
}
