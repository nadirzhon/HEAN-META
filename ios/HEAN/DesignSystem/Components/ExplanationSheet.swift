//
//  ExplanationSheet.swift
//  HEAN
//
//  Educational bottom sheet for explaining trading terms
//

import SwiftUI

struct ExplanationSheet: View {
    let term: ExplanationTerm
    let currentValue: Double?
    @SwiftUI.Environment(\.dismiss) private var dismiss: DismissAction

    init(term: ExplanationTerm, currentValue: Double?) {
        self.term = term
        self.currentValue = currentValue
    }

    private var entry: ExplanationEntry {
        ExplanationDatabase.entry(for: term)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // What is it
                    sectionCard(header: "Что это?", content: entry.whatIsIt)

                    // How to read
                    sectionCard(header: "Как читать?", content: entry.howToRead)

                    // Current meaning
                    if let value = currentValue {
                        sectionCard(
                            header: "Сейчас",
                            content: contextualMeaning(value)
                        )
                    }

                    // Warning
                    if let warning = entry.warning {
                        warningCard(warning)
                    }

                    // Formula
                    if let formula = entry.formula {
                        sectionCard(header: "Формула", content: formula)
                    }
                }
                .padding()
            }
            .background(Theme.Colors.background.ignoresSafeArea())
            .navigationTitle(entry.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Готово") { dismiss() }
                        .foregroundColor(Theme.Colors.accent)
                }
            }
        }
    }

    private func sectionCard(header: String, content: String) -> some View {
        GlassCard(padding: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text(header)
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(Theme.Colors.textSecondary)
                Text(content)
                    .font(.subheadline)
                    .foregroundColor(Theme.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func warningCard(_ warning: String) -> some View {
        GlassCard(padding: 16) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(Theme.Colors.warning)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Внимание")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(Theme.Colors.warning)
                    Text(warning)
                        .font(.subheadline)
                        .foregroundColor(Theme.Colors.textPrimary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func contextualMeaning(_ value: Double) -> String {
        switch term {
        case .temperature:
            return ExplanationDatabase.temperatureMeaning(value)
        case .entropy:
            return ExplanationDatabase.entropyMeaning(value)
        default:
            return entry.currentMeaning(value)
        }
    }
}
