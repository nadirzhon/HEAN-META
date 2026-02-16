//
//  GenesisView.swift
//  HEAN
//
//  Interactive cinematic tour: "The Birth of HEAN Consciousness"
//  5 scenes with TabView paging, hero orb morph, and synchronized text animations.
//

import SwiftUI

struct GenesisView: View {
    @State private var currentScene = 0
    @State private var isAutoPlaying = false
    @State private var autoPlayTask: Task<Void, Never>?
    @State private var titleVisible = false
    @Namespace private var heroNamespace

    private let totalScenes = 5
    private let sceneDuration: Double = 8.0

    var body: some View {
        ZStack {
            Theme.Colors.background
                .ignoresSafeArea()

            // Layer 1: Scene pages (swipeable)
            TabView(selection: $currentScene) {
                GenesisScene1View(sceneActive: currentScene == 0)
                    .tag(0)
                GenesisScene2View(sceneActive: currentScene == 1)
                    .tag(1)
                GenesisScene3View(sceneActive: currentScene == 2)
                    .tag(2)
                GenesisScene4View(sceneActive: currentScene == 3)
                    .tag(3)
                GenesisScene5View(sceneActive: currentScene == 4)
                    .tag(4)
            }
            .tabViewStyle(.page(indexDisplayMode: .never))

            // Layer 2: Hero morph overlay (floats above pages)
            GenesisHeroLayer(currentScene: currentScene)

            // Layer 3: Title + Navigation
            VStack {
                // Scene title (animated character-by-character)
                HStack {
                    AnimatedText(
                        sceneTitle,
                        isVisible: titleVisible,
                        font: Theme.Typography.caption(12, weight: .medium),
                        color: Theme.Colors.textTertiary,
                        stagger: 0.04
                    )
                    .textCase(.uppercase)
                    .id(currentScene) // Reset animation on scene change

                    Spacer()

                    Text("\(currentScene + 1)/\(totalScenes)")
                        .font(Theme.Typography.monoFont(12))
                        .foregroundColor(Theme.Colors.textTertiary)
                }
                .padding(.horizontal, Theme.Spacing.xl)
                .padding(.top, Theme.Spacing.md)

                Spacer()

                // Navigation bar
                GenesisNavBar(
                    currentScene: $currentScene,
                    totalScenes: totalScenes,
                    isAutoPlaying: $isAutoPlaying,
                    namespace: heroNamespace,
                    onPrevious: previousScene,
                    onNext: nextScene,
                    onToggleAutoPlay: toggleAutoPlay
                )
                .padding(.horizontal, Theme.Spacing.xl)
                .padding(.bottom, Theme.Spacing.xxl)
            }
        }
        .onChange(of: currentScene) { _, _ in
            // Haptic feedback on scene change
            Haptics.light()

            // Re-trigger title animation
            titleVisible = false
            withAnimation(.easeOut(duration: 0.3).delay(0.2)) {
                titleVisible = true
            }
        }
        .onAppear {
            withAnimation(.easeOut(duration: 0.5).delay(0.3)) {
                titleVisible = true
            }
        }
        .onDisappear {
            autoPlayTask?.cancel()
        }
    }

    private var sceneTitle: String {
        switch currentScene {
        case 0: return L.isRussian ? "ГЛАВА I" : "CHAPTER I"
        case 1: return L.isRussian ? "ГЛАВА II" : "CHAPTER II"
        case 2: return L.isRussian ? "ГЛАВА III" : "CHAPTER III"
        case 3: return L.isRussian ? "ГЛАВА IV" : "CHAPTER IV"
        case 4: return L.isRussian ? "ГЛАВА V" : "CHAPTER V"
        default: return ""
        }
    }

    private func nextScene() {
        guard currentScene < totalScenes - 1 else { return }
        withAnimation(.easeInOut(duration: 0.4)) {
            currentScene += 1
        }
    }

    private func previousScene() {
        guard currentScene > 0 else { return }
        withAnimation(.easeInOut(duration: 0.4)) {
            currentScene -= 1
        }
    }

    private func toggleAutoPlay() {
        isAutoPlaying.toggle()
        Haptics.selection()
        if isAutoPlaying {
            startAutoPlay()
        } else {
            autoPlayTask?.cancel()
        }
    }

    private func startAutoPlay() {
        autoPlayTask?.cancel()
        autoPlayTask = Task {
            while !Task.isCancelled && isAutoPlaying {
                try? await Task.sleep(for: .seconds(sceneDuration))
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    if currentScene < totalScenes - 1 {
                        withAnimation(.easeInOut(duration: 0.6)) {
                            currentScene += 1
                        }
                    } else {
                        isAutoPlaying = false
                    }
                }
            }
        }
    }
}

#Preview {
    GenesisView()
        .preferredColorScheme(.dark)
}
