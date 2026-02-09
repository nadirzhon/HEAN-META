//
//  ExplainableModifier.swift
//  HEAN
//
//  Long-press any view to get an explanation
//

import SwiftUI

struct ExplainableModifier: ViewModifier {
    let term: ExplanationTerm
    let value: Double?
    @State private var showSheet = false

    func body(content: Content) -> some View {
        content
            .onLongPressGesture {
                Haptics.light()
                showSheet = true
            }
            .sheet(isPresented: $showSheet) {
                ExplanationSheet(term: term, currentValue: value)
                    .presentationDetents(Set<PresentationDetent>([.medium, .large]))
            }
    }
}

extension View {
    func explainable(_ term: ExplanationTerm, value: Double? = nil) -> some View {
        modifier(ExplainableModifier(term: term, value: value))
    }
}
