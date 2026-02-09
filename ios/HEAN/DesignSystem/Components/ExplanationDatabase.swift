//
//  ExplanationDatabase.swift
//  HEAN
//
//  Static educational content for market terms
//

import Foundation

enum ExplanationTerm: String, CaseIterable {
    case temperature
    case entropy
    case phase
    case szilardProfit
    case buyPressure
    case sellPressure
    case moneyFlowDelta
    case whaleTrades
    case liquidations
    case mmActivity
    case institutionalFlow
    case retailSentiment
    case whaleActivity
    case arbPressure
    case riskState
    case killswitch
    case drawdown
    case winRate
    case riskRewardRatio
    case profitFactor
}

struct ExplanationEntry {
    let title: String
    let whatIsIt: String
    let howToRead: String
    let warning: String?
    let formula: String?

    func currentMeaning(_ value: Double) -> String {
        ExplanationDatabase.meaning(for: self, value: value)
    }
}

enum ExplanationDatabase {

    static func entry(for term: ExplanationTerm) -> ExplanationEntry {
        switch term {
        case .temperature:
            return ExplanationEntry(
                title: "Температура рынка",
                whatIsIt: "Показывает насколько активен рынок прямо сейчас. Чем больше ордеров и сделок — тем горячее.",
                howToRead: "0-200: Холодно. Боковик, скучно.\n200-600: Тепло. Нормальный рынок.\n600-1200: Горячо. Сильное движение.\n1200+: Экстрим. Паника или эйфория.",
                warning: nil,
                formula: "T = кинетическая энергия ордербука (Больцман)"
            )
        case .entropy:
            return ExplanationEntry(
                title: "Энтропия рынка",
                whatIsIt: "Измеряет хаос или порядок в движениях цены. Низкая энтропия = рынок сжат как пружина. Высокая = рынок расслаблен.",
                howToRead: "0-0.3: Сильный порядок. Пружина сжата.\n0.3-0.7: Переходное состояние.\n0.7-1.0: Равновесие. Энергия рассеяна.",
                warning: "Низкая энтропия часто предшествует резкому движению!",
                formula: "S = -Σ p(i) × ln(p(i))"
            )
        case .phase:
            return ExplanationEntry(
                title: "Фаза рынка",
                whatIsIt: "Рынок как вода — может быть в трёх состояниях: лёд (тихо), вода (нормально), пар (бурлит).",
                howToRead: "Лёд: Низкая волатильность. Не торгуй.\nВода: Нормальная активность. Можно торговать.\nПар: Высокая волатильность. Осторожно!",
                warning: "Переходы между фазами — самые опасные и прибыльные моменты.",
                formula: nil
            )
        case .szilardProfit:
            return ExplanationEntry(
                title: "Прибыль Сциларда",
                whatIsIt: "Теоретическая прибыль от использования информационного преимущества. Показывает сколько можно заработать на текущем информационном дисбалансе.",
                howToRead: "Чем выше значение — тем больше информационная асимметрия на рынке и потенциал для прибыли.",
                warning: nil,
                formula: "W = kT × ln(2) × информационный бит"
            )
        case .buyPressure:
            return ExplanationEntry(
                title: "Давление покупателей",
                whatIsIt: "Показывает силу покупателей в ордербуке. Чем больше ордеров на покупку — тем сильнее давление вверх.",
                howToRead: "Высокое значение = покупатели доминируют.\nСравни с давлением продавцов для полной картины.",
                warning: "Иногда крупные игроки ставят фейковые ордера (спуфинг). Смотри также дельту — она показывает реальные сделки.",
                formula: nil
            )
        case .sellPressure:
            return ExplanationEntry(
                title: "Давление продавцов",
                whatIsIt: "Показывает силу продавцов в ордербуке. Много ордеров на продажу = давление вниз.",
                howToRead: "Высокое значение = продавцы доминируют.\nСравни с давлением покупателей.",
                warning: nil,
                formula: nil
            )
        case .moneyFlowDelta:
            return ExplanationEntry(
                title: "Поток денег (Дельта)",
                whatIsIt: "Разница между объёмом покупок и продаж за период. Положительная дельта = больше покупок. Отрицательная = больше продаж.",
                howToRead: "1 мин: Мгновенный импульс.\n5 мин: Краткосрочный тренд.\n15 мин: Устойчивое направление.",
                warning: nil,
                formula: "Δ = Σ(покупки) - Σ(продажи)"
            )
        case .whaleTrades:
            return ExplanationEntry(
                title: "Сделки китов",
                whatIsIt: "Крупные сделки (обычно >2 BTC). Киты — это крупные игроки, фонды и институционалы. Их действия часто определяют направление рынка.",
                howToRead: "Серия крупных покупок = накопление.\nОдна большая продажа = фиксация прибыли или начало распродажи.",
                warning: "Одна крупная продажа на растущем рынке может быть ложной тревогой — кит просто фиксирует прибыль.",
                formula: nil
            )
        case .liquidations:
            return ExplanationEntry(
                title: "Ликвидации",
                whatIsIt: "Принудительное закрытие позиций биржей когда убыток превышает маржу. Ликвидации лонг = стопы покупателей. Ликвидации шорт = стопы продавцов.",
                howToRead: "Большие ликвидации лонг = покупатели уже получили стопы → меньше давления вниз.\nБольшие ликвидации шорт = продавцы сожжены.",
                warning: "Каскад ликвидаций может вызвать резкое движение цены в одном направлении.",
                formula: nil
            )
        case .mmActivity:
            return ExplanationEntry(
                title: "Маркетмейкеры",
                whatIsIt: "Профессиональные участники которые поддерживают ликвидность — одновременно ставят ордера на покупку и продажу, зарабатывая на спреде.",
                howToRead: "Высокая активность = узкий спред, стабильный рынок.\nНизкая активность = ММ уходят, возможен резкий рывок.",
                warning: "Когда ММ убирают ордера — жди сильного движения.",
                formula: nil
            )
        case .institutionalFlow:
            return ExplanationEntry(
                title: "Институционалы",
                whatIsIt: "Крупные фонды, банки, компании. Торгуют большими объёмами, часто через iceberg-ордера (разбивают большой ордер на мелкие чтобы не привлекать внимание).",
                howToRead: "Высокий поток = институционал активно набирает/сбрасывает позицию.\nНаправление потока показывает куда смотрят 'умные деньги'.",
                warning: nil,
                formula: nil
            )
        case .retailSentiment:
            return ExplanationEntry(
                title: "Розница (Ритейл)",
                whatIsIt: "Обычные трейдеры — ты и я. Часто покупают на хаях и продают на лоях. Настроение розницы — хороший контр-индикатор.",
                howToRead: "Высокий оптимизм = все уже купили → кто будет покупать дальше?\nВысокий страх = все продали → потенциал для разворота.",
                warning: "Если все ритейл-трейдеры лонгят — крупные игроки могут собрать их стопы.",
                formula: nil
            )
        case .whaleActivity:
            return ExplanationEntry(
                title: "Активность китов",
                whatIsIt: "Крупные держатели с большими балансами. Их перемещения средств часто предсказывают крупные движения цены.",
                howToRead: "Высокая активность = киты двигаются.\nВажно смотреть направление — покупают или продают.",
                warning: nil,
                formula: nil
            )
        case .arbPressure:
            return ExplanationEntry(
                title: "Арбитражные боты",
                whatIsIt: "Автоматические программы которые ищут разницу цен между биржами и зарабатывают на ней. Их активность выравнивает цены между площадками.",
                howToRead: "Высокая активность = большие расхождения цен → рынок нестабилен.\nНизкая активность = цены синхронны, рынок в равновесии.",
                warning: nil,
                formula: nil
            )
        case .riskState:
            return ExplanationEntry(
                title: "Состояние риска",
                whatIsIt: "Система управления риском защищает твой капитал. Она может быть в нескольких состояниях — от нормального до полной остановки.",
                howToRead: "NORMAL: Всё в порядке, торговля разрешена.\nSOFT_BRAKE: Предупреждение, ограничение новых позиций.\nQUARANTINE: Символ изолирован из-за проблем.\nHARD_STOP: Полная остановка торговли.",
                warning: "Если система перешла в HARD_STOP — не пытайся обойти. Она защищает твои деньги.",
                formula: nil
            )
        case .killswitch:
            return ExplanationEntry(
                title: "Kill Switch",
                whatIsIt: "Аварийный выключатель. Срабатывает при просадке >20% от начального капитала. Закрывает все позиции и останавливает торговлю.",
                howToRead: "Зелёный = не активен, всё в порядке.\nКрасный = сработал! Торговля остановлена.",
                warning: "Kill Switch — последняя линия защиты. Если он сработал, значит что-то пошло серьёзно не так.",
                formula: nil
            )
        case .drawdown:
            return ExplanationEntry(
                title: "Просадка",
                whatIsIt: "Снижение баланса от максимума. Если баланс был $1000 и стал $900 — просадка 10%.",
                howToRead: "0-5%: Нормальная рабочая просадка.\n5-10%: Повышенная, будь внимателен.\n10-20%: Серьёзная, система ограничит торговлю.\n20%+: Kill Switch сработает.",
                warning: nil,
                formula: "DD = (Пик - Текущий) / Пик × 100%"
            )
        case .winRate:
            return ExplanationEntry(
                title: "Процент побед",
                whatIsIt: "Доля прибыльных сделок. Win rate 60% значит 6 из 10 сделок закрылись в плюс.",
                howToRead: "40-50%: Нормально если R:R > 2.\n50-60%: Хорошо.\n60%+: Отлично.\nВажен не только win rate, но и R:R.",
                warning: nil,
                formula: "WR = Прибыльные / Всего × 100%"
            )
        case .riskRewardRatio:
            return ExplanationEntry(
                title: "Risk/Reward (R:R)",
                whatIsIt: "Соотношение потенциальной прибыли к потенциальному убытку. R:R 1:2 значит что ты рискуешь $1 чтобы заработать $2.",
                howToRead: "1:1: Минимум. Нужен win rate > 55%.\n1:2: Хорошо. Достаточно win rate 40%.\n1:3+: Отлично. Можно ошибаться чаще.",
                warning: nil,
                formula: "R:R = (TP - Entry) / (Entry - SL)"
            )
        case .profitFactor:
            return ExplanationEntry(
                title: "Профит-фактор",
                whatIsIt: "Отношение общей прибыли к общему убытку. Профит-фактор > 1 = система зарабатывает больше чем теряет.",
                howToRead: "< 1.0: Система убыточна.\n1.0-1.5: Слабо прибыльная.\n1.5-2.0: Хорошая система.\n2.0+: Отличная система.",
                warning: nil,
                formula: "PF = Σ(прибыль) / Σ(убытки)"
            )
        }
    }

    static func meaning(for entry: ExplanationEntry, value: Double) -> String {
        // Generic fallback — specific views can provide their own context
        "Текущее значение: \(String(format: "%.2f", value))"
    }

    static func temperatureMeaning(_ value: Double) -> String {
        if value < 200 { return "Сейчас \(Int(value)) — рынок холодный. Мало активности, не лучшее время для торговли." }
        if value < 600 { return "Сейчас \(Int(value)) — рынок тёплый. Нормальная активность, можно искать сигналы." }
        if value < 1200 { return "Сейчас \(Int(value)) — рынок горячий! Сильное движение, будь внимателен." }
        return "Сейчас \(Int(value)) — экстремальная активность! Паника или эйфория."
    }

    static func entropyMeaning(_ value: Double) -> String {
        if value < 0.3 { return "Энтропия \(String(format: "%.2f", value)) — сильный порядок. Пружина сжата, возможен резкий рывок." }
        if value < 0.7 { return "Энтропия \(String(format: "%.2f", value)) — переходное состояние." }
        return "Энтропия \(String(format: "%.2f", value)) — равновесие. Энергия рассеяна."
    }

    static func phaseMeaning(_ phase: String) -> String {
        switch phase.lowercased() {
        case "ice": return "Фаза: Лёд. Рынок спит, низкая волатильность. Подожди."
        case "water": return "Фаза: Вода. Нормальная активность, можно торговать."
        case "vapor": return "Фаза: Пар. Высокая волатильность, будь осторожен!"
        default: return "Фаза: \(phase.capitalized)"
        }
    }
}
