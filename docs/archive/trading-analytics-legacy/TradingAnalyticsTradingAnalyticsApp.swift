import SwiftUI

@main
struct TradingAnalyticsApp: App {
    @StateObject private var dataManager = TradingDataManager()
    @StateObject private var aiAssistant = AITradingAssistant()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(dataManager)
                .environmentObject(aiAssistant)
        }
    }
}
