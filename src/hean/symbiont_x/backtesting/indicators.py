"""
Technical analysis indicators library
"""

def calculate_sma(data: list[float], period: int) -> list[float | None]:
    """Calculates Simple Moving Average (SMA)"""
    if not data or period <= 0 or len(data) < period:
        return [None] * len(data)

    sma_values = [None] * (period - 1)

    # Calculate first SMA
    current_sum = sum(data[:period])
    sma_values.append(current_sum / period)

    # Efficiently calculate subsequent SMAs
    for i in range(period, len(data)):
        current_sum += data[i] - data[i - period]
        sma_values.append(current_sum / period)

    return sma_values

def calculate_ema(data: list[float], period: int) -> list[float | None]:
    """Calculates Exponential Moving Average (EMA)"""
    if not data or period <= 0 or len(data) < period:
        return [None] * len(data)

    ema_values = [None] * (period - 1)

    # First EMA is the SMA of the first period
    first_sma = sum(data[:period]) / period
    ema_values.append(first_sma)

    multiplier = 2 / (period + 1)

    # Calculate subsequent EMAs
    for i in range(period, len(data)):
        prev_ema = ema_values[-1]
        if prev_ema is None: # Should not happen after first calculation
             ema_values.append(None)
             continue

        new_ema = (data[i] - prev_ema) * multiplier + prev_ema
        ema_values.append(new_ema)

    return ema_values

def calculate_rsi(data: list[float], period: int = 14) -> list[float | None]:
    """Calculates Relative Strength Index (RSI)"""
    if not data or period <= 0 or len(data) < period:
        return [None] * len(data)

    rsi_values = [None] * period
    gains = []
    losses = []

    # Calculate initial average gain and loss
    for i in range(1, period + 1):
        change = data[i] - data[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        rs = float('inf')
    else:
        rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))
    rsi_values.append(rsi)

    # Calculate subsequent RSI
    for i in range(period + 1, len(data)):
        change = data[i] - data[i - 1]
        gain = change if change > 0 else 0
        loss = abs(change) if change < 0 else 0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rs = float('inf')
        else:
            rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)

    return rsi_values
