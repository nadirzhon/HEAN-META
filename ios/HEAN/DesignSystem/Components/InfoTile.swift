//
//  InfoTile.swift
//  HEAN
//
//  Tappable metric card with optional explanation sheet
//

import SwiftUI

struct InfoTile: View {
    let icon: String
    let title: String
    let value: String
    let subtitle: String?
    let color: Color
    let term: ExplanationTerm?
    let numericValue: Double?

    @State private var showExplanation = false

    init(
        icon: String,
        title: String,
        value: String,
        subtitle: String? = nil,
        color: Color = Theme.Colors.accent,
        term: ExplanationTerm? = nil,
        numericValue: Double? = nil
    ) {
        self.icon = icon
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.color = color
        self.term = term
        self.numericValue = numericValue
    }

    var body: some View {
        Button {
            if term != nil {
                Haptics.light()
                showExplanation = true
            }
        } label: {
            GlassCard(padding: 10) {
                VStack(spacing: 6) {
                    Image(systemName: icon)
                        .font(.system(size: 18))
                        .foregroundColor(color)

                    Text(title)
                        .font(.caption2)
                        .fontWeight(.medium)
                        .foregroundColor(Theme.Colors.textSecondary)
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)

                    Text(value)
                        .font(.system(.subheadline, design: .monospaced))
                        .fontWeight(.bold)
                        .foregroundColor(Theme.Colors.textPrimary)
                        .lineLimit(1)
                        .minimumScaleFactor(0.6)

                    if let subtitle = subtitle {
                        Text(subtitle)
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.textTertiary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.7)
                    }
                }
                .frame(maxWidth: .infinity)
            }
        }
        .buttonStyle(.plain)
        .sheet(isPresented: $showExplanation) {
            if let term = term {
                ExplanationSheet(term: term, currentValue: numericValue)
                    .presentationDetents(Set<PresentationDetent>([.medium, .large]))
            }
        }
    }
}
