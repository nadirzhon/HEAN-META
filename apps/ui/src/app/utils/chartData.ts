export interface CandleData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OrderBlock {
  price: number;
  label: string;
  type: "support" | "resistance" | "liquidity";
}

export interface DecisionMarker {
  timestamp: number;
  price: number;
  type: "entry" | "exit";
  side: "LONG" | "SHORT";
  reason: string;
  confidence: number;
  suppressedAlternatives?: string[];
}

export interface ProfitZone {
  startTime: number;
  endTime: number;
  priceRange: { low: number; high: number };
  type: "optimal" | "weak" | "risk-dominant";
  label: string;
}

// Generate realistic candlestick data
export function generateCandleData(count: number, basePrice: number = 43000): CandleData[] {
  const data: CandleData[] = [];
  let currentPrice = basePrice;
  const now = Date.now();
  
  for (let i = 0; i < count; i++) {
    const timestamp = now - (count - i) * 60000; // 1-minute candles
    
    // Random walk with trend
    const trend = Math.sin(i / 20) * 50;
    const volatility = Math.random() * 100 - 50;
    currentPrice = currentPrice + trend + volatility;
    
    const open = currentPrice;
    const close = open + (Math.random() * 200 - 100);
    const high = Math.max(open, close) + Math.random() * 50;
    const low = Math.min(open, close) - Math.random() * 50;
    const volume = Math.random() * 1000000 + 500000;
    
    data.push({
      timestamp,
      open,
      high,
      low,
      close,
      volume,
    });
    
    currentPrice = close;
  }
  
  return data;
}

// Generate decision markers
export function generateDecisionMarkers(candles: CandleData[]): DecisionMarker[] {
  const markers: DecisionMarker[] = [];
  const markerIndices = [20, 45, 78, 110, 145];
  
  markerIndices.forEach((idx, i) => {
    if (idx < candles.length) {
      const candle = candles[idx];
      const isEntry = i % 2 === 0;
      const isLong = i % 3 !== 0;
      
      markers.push({
        timestamp: candle.timestamp,
        price: isEntry ? candle.close : candle.close + (isLong ? 200 : -200),
        type: isEntry ? "entry" : "exit",
        side: isLong ? "LONG" : "SHORT",
        reason: isEntry 
          ? (isLong ? "Impulse aligned, risk allowed" : "Mean reversion setup")
          : (Math.random() > 0.5 ? "TP_HIT" : "TIMEOUT_TTL"),
        confidence: 70 + Math.random() * 25,
        suppressedAlternatives: isEntry ? ["Volume too low", "Max positions reached"] : undefined,
      });
    }
  });
  
  return markers;
}

// Generate profit zones
export function generateProfitZones(candles: CandleData[]): ProfitZone[] {
  const zones: ProfitZone[] = [];
  const zoneCount = 4;
  const chunkSize = Math.floor(candles.length / zoneCount);
  
  const types: Array<"optimal" | "weak" | "risk-dominant"> = ["optimal", "weak", "risk-dominant", "optimal"];
  const labels = [
    "Optimal execution zone",
    "Low expectancy area",
    "Exit-dominant area",
    "High conviction zone"
  ];
  
  for (let i = 0; i < zoneCount; i++) {
    const startIdx = i * chunkSize;
    const endIdx = Math.min((i + 1) * chunkSize, candles.length - 1);
    
    const startCandle = candles[startIdx];
    const endCandle = candles[endIdx];
    
    const prices = candles.slice(startIdx, endIdx + 1).map(c => [c.low, c.high]).flat();
    const low = Math.min(...prices);
    const high = Math.max(...prices);
    
    zones.push({
      startTime: startCandle.timestamp,
      endTime: endCandle.timestamp,
      priceRange: { low, high },
      type: types[i],
      label: labels[i],
    });
  }
  
  return zones;
}

// Generate order blocks
export function generateOrderBlocks(candles: CandleData[]): OrderBlock[] {
  if (candles.length === 0) return [];
  
  const prices = candles.map(c => [c.low, c.high]).flat();
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const midPrice = (minPrice + maxPrice) / 2;
  
  return [
    { price: maxPrice - 100, label: "Resistance", type: "resistance" },
    { price: midPrice, label: "Liquidity Zone", type: "liquidity" },
    { price: minPrice + 100, label: "Support", type: "support" },
  ];
}
