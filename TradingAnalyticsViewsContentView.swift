import SwiftUI

// MARK: - New Design System
struct Theme {
    static let background = Color(red: 0.97, green: 0.97, blue: 0.98) // #F7F7F8
    static let text = Color(red: 0.12, green: 0.12, blue: 0.12) // #1E1E1E
    static let accent = Color.blue
    static let secondaryText = Color.gray
}

struct ContentView: View {
    @State private var selectedTab: Tab = .dashboard
    
    enum Tab {
        case dashboard
        case analytics
        case trades
        case ai
    }
    
    var body: some View {
        NavigationStack {
            TabView(selection: $selectedTab) {
                DashboardView()
                    .tabItem {
                        Label("Dashboard", systemImage: selectedTab == .dashboard ? "chart.bar.fill" : "chart.bar")
                    }
                    .tag(Tab.dashboard)
                
                AnalyticsView()
                    .tabItem {
                        Label("Analytics", systemImage: selectedTab == .analytics ? "chart.line.uptrend.xyaxis.circle.fill" : "chart.line.uptrend.xyaxis.circle")
                    }
                    .tag(Tab.analytics)
                
                TradesListView()
                    .tabItem {
                        Label("Trades", systemImage: selectedTab == .trades ? "list.bullet.rectangle.fill" : "list.bullet.rectangle")
                    }
                    .tag(Tab.trades)
                
                AIAssistantView()
                    .tabItem {
                        Label("AI", systemImage: selectedTab == .ai ? "sparkles" : "sparkles")
                    }
                    .tag(Tab.ai)
            }
            .accentColor(Theme.accent)
            .navigationTitle(navigationTitle)
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    RefreshButton()
                }
            }
            .background(Theme.background)
        }
        .task {
            // Data loading logic remains the same
        }
    }
    
    private var navigationTitle: String {
        switch selectedTab {
        case .dashboard: return "Dashboard"
        case .analytics: return "Analytics"
        case .trades: return "Trades"
        case .ai: return "AI Assistant"
        }
    }
}

// MARK: - Refresh Button (Adapted for New Theme)

struct RefreshButton: View {
    // EnvironmentObject and State remain the same
    @EnvironmentObject var dataManager: TradingDataManager
    @State private var isRotating = false
    
    var body: some View {
        Button {
            Task {
                withAnimation(.linear(duration: 1).repeatCount(3)) {
                    isRotating = true
                }
                await dataManager.loadBacktestData()
                isRotating = false
            }
        } label: {
            Image(systemName: "arrow.clockwise")
                .font(.system(size: 16, weight: .semibold))
                .foregroundColor(Theme.accent)
                .rotationEffect(.degrees(isRotating ? 360 : 0))
        }
        .disabled(dataManager.isLoading)
    }
}

// MARK: - Preview (Updated for New Design)

#Preview {
    ContentView()
        .environmentObject(TradingDataManager())
        .environmentObject(AITradingAssistant())
}
