# 🚀 HEAN Quick Start Checklist

**Status**: Docker build in progress...
**Mode**: TESTNET/PAPER (Safe)
**Date**: 2026-01-27

## ✅ Verification Commands (Run After Build)

```bash
# 1. Health
curl http://localhost:8000/health

# 2. Multi-symbol check
curl http://localhost:8000/system/v1/dashboard | jq '.active_symbols'

# 3. Strategies (expect 3)
curl http://localhost:8000/strategies | jq 'length'

# 4. C++ status
curl http://localhost:8000/system/cpp/status | jq

# 5. UI
open http://localhost:3000

# 6. Logs (first 50 lines)
docker-compose logs api | head -50
```

## 🎯 Expected Results

- Health: `{"status": "healthy"}`
- Symbols: `["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]`
- Strategies: `3`
- C++ available: `false` (fallback to Python)
- UI: Shows PAPER mode, 5 symbols, WebSocket connected

## 🚨 Red Flags (Should NOT See)

- ❌ "AGGRESSIVE MODE" in logs
- ❌ "bypass" in logs  
- ❌ Killswitch triggered
- ❌ Container restart loops

See `FINAL_DELIVERY_REPORT.md` for complete guide.
