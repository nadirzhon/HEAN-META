import SwiftUI

struct Theme {
    static let background = Color(red: 0.97, green: 0.97, blue: 0.98) // #F7F7F8
    static let text = Color(red: 0.12, green: 0.12, blue: 0.12) // #1E1E1E
    static let accent = Color.blue
    static let secondaryText = Color.gray
    static let cardBackground = Color.white
    static let separator = Color(white: 0.85)
    
    static let positive = Color.green
    static let negative = Color.red
}

struct CardViewModifier: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(20)
            .background(Theme.cardBackground)
            .cornerRadius(24)
            .shadow(color: Color.black.opacity(0.05), radius: 10, x: 0, y: 5)
    }
}

extension View {
    func card() -> some View {
        self.modifier(CardViewModifier())
    }
}
