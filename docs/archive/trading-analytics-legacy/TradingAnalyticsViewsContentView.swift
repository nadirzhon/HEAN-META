import SwiftUI

struct ContentView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    @EnvironmentObject var aiAssistant: AITradingAssistant
    @State private var selectedTab: Tab = .dashboard
    
    enum Tab {
        case dashboard
        case analytics
        case trades
        case ai
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                // Динамический градиентный фон
                AnimatedGradientBackground()
                    .ignoresSafeArea()
                
                VStack(spacing: 0) {
                    // Главный контент
                    TabView(selection: $selectedTab) {
                        DashboardView()
                            .tag(Tab.dashboard)
                        
                        AnalyticsView()
                            .tag(Tab.analytics)
                        
                        TradesListView()
                            .tag(Tab.trades)
                        
                        AIAssistantView()
                            .tag(Tab.ai)
                    }
                    .tabViewStyle(.page(indexDisplayMode: .never))
                    
                    // Кастомная навигационная панель с Liquid Glass
                    GlassTabBar(selectedTab: $selectedTab)
                        .padding(.horizontal)
                        .padding(.bottom, 8)
                }
            }
            .navigationTitle(navigationTitle)
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    RefreshButton()
                }
            }
        }
        .task {
            await dataManager.loadBacktestData()
            if let results = dataManager.currentBacktest {
                await aiAssistant.analyzeBacktestResults(results)
            }
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

// MARK: - Animated Gradient Background

struct AnimatedGradientBackground: View {
    @State private var animate = false
    
    var body: some View {
        LinearGradient(
            colors: [
                Color(red: 0.1, green: 0.1, blue: 0.2),
                Color(red: 0.15, green: 0.15, blue: 0.3),
                Color(red: 0.1, green: 0.15, blue: 0.25)
            ],
            startPoint: animate ? .topLeading : .bottomLeading,
            endPoint: animate ? .bottomTrailing : .topTrailing
        )
        .onAppear {
            withAnimation(.easeInOut(duration: 8).repeatForever(autoreverses: true)) {
                animate.toggle()
            }
        }
    }
}

// MARK: - Glass Tab Bar

struct GlassTabBar: View {
    @Binding var selectedTab: ContentView.Tab
    @Namespace private var namespace
    
    var body: some View {
        GlassEffectContainer(spacing: 12) {
            HStack(spacing: 12) {
                TabButton(
                    icon: "chart.bar.fill",
                    title: "Dashboard",
                    isSelected: selectedTab == .dashboard,
                    namespace: namespace
                ) {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                        selectedTab = .dashboard
                    }
                }
                
                TabButton(
                    icon: "chart.line.uptrend.xyaxis",
                    title: "Analytics",
                    isSelected: selectedTab == .analytics,
                    namespace: namespace
                ) {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                        selectedTab = .analytics
                    }
                }
                
                TabButton(
                    icon: "list.bullet.rectangle",
                    title: "Trades",
                    isSelected: selectedTab == .trades,
                    namespace: namespace
                ) {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                        selectedTab = .trades
                    }
                }
                
                TabButton(
                    icon: "sparkles",
                    title: "AI",
                    isSelected: selectedTab == .ai,
                    namespace: namespace
                ) {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                        selectedTab = .ai
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 12)
        }
        .glassEffect(.regular.interactive())
    }
}

struct TabButton: View {
    let icon: String
    let title: String
    let isSelected: Bool
    let namespace: Namespace.ID
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 20, weight: .semibold))
                    .symbolEffect(.bounce, value: isSelected)
                
                Text(title)
                    .font(.caption2)
                    .fontWeight(.medium)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .foregroundStyle(isSelected ? Color.white : Color.white.opacity(0.6))
            .background {
                if isSelected {
                    Capsule()
                        .fill(Color.white.opacity(0.2))
                        .matchedGeometryEffect(id: "selectedTab", in: namespace)
                }
            }
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Refresh Button

struct RefreshButton: View {
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
                .rotationEffect(.degrees(isRotating ? 360 : 0))
        }
        .disabled(dataManager.isLoading)
    }
}

#Preview {
    ContentView()
        .environmentObject(TradingDataManager())
        .environmentObject(AITradingAssistant())
}
