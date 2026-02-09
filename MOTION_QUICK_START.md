# HEAN Motion System - Quick Start

Get started with HEAN animations in 5 minutes.

---

## Installation

Already installed. Just import:

```typescript
import { GlassCard, ConnectionStateIndicator, motionTokens } from '@/app/motion';
```

---

## 1. Basic Glass Card

```typescript
import { GlassCard } from '@/app/motion';

function MyComponent() {
  return (
    <GlassCard hoverable>
      <div className="p-6">
        <h2>Portfolio</h2>
        <p>Content here</p>
      </div>
    </GlassCard>
  );
}
```

**Result:** Interactive card with hover lift, glow, and glass effect.

---

## 2. WebSocket Status

```typescript
import { ConnectionStateIndicator } from '@/app/motion';

function StatusBar({ wsStatus }) {
  return (
    <div className="flex items-center gap-2">
      <ConnectionStateIndicator state={wsStatus} showLabel />
    </div>
  );
}
```

**Props:** `state: 'connected' | 'reconnecting' | 'disconnected'`

---

## 3. Animated Numbers

```typescript
import { AnimatedCurrency } from '@/app/components/animated/AnimatedNumber';

function Balance({ equity }) {
  return <AnimatedCurrency value={equity} flashOnChange />;
}
```

**Result:** Numbers count up/down smoothly with flash on change.

---

## 4. Risk State Borders

```typescript
import { RiskStateBorder } from '@/app/motion';

function RiskPanel({ riskState }) {
  return (
    <RiskStateBorder state={riskState}>
      <div className="p-4">Risk metrics</div>
    </RiskStateBorder>
  );
}
```

**States:**
- `normal` → Cyan border
- `softBrake` → Amber border
- `quarantine` → Orange + pulse
- `hardStop` → Red + strong pulse

---

## 5. Event Flashes

```typescript
import { EventFlash, PriceFlash } from '@/app/motion';

function PriceDisplay({ current, previous }) {
  return (
    <div className="relative">
      <PriceFlash value={current} previousValue={previous} />
      <span>${current}</span>
    </div>
  );
}
```

**Result:** Brief background flash (green/red) on price change.

---

## 6. Interactive Buttons

```typescript
import { RippleButton } from '@/app/motion';

function ControlPanel() {
  return (
    <RippleButton variant="primary" onClick={handleStart}>
      Start Trading
    </RippleButton>
  );
}
```

**Variants:**
- `default` - White/gray
- `primary` - Cyan accent
- `danger` - Red accent

---

## 7. Toggles & Checkboxes

```typescript
import { AnimatedToggle, AnimatedCheckbox } from '@/app/motion';

function Settings() {
  const [enabled, setEnabled] = useState(false);

  return (
    <>
      <AnimatedToggle
        enabled={enabled}
        onChange={setEnabled}
        label="Enable Auto-Trading"
      />

      <AnimatedCheckbox
        checked={agreed}
        onChange={setAgreed}
        label="I understand the risks"
      />
    </>
  );
}
```

---

## 8. Loading States

```typescript
import { LoadingDots, SkeletonLoader } from '@/app/motion';

function LoadingView() {
  return (
    <div>
      <LoadingDots />
      <SkeletonLoader width="100%" height={20} />
      <SkeletonLoader width={200} height={40} />
    </div>
  );
}
```

---

## 9. Modals/Dialogs

```typescript
import { motion, AnimatePresence } from 'framer-motion';
import { motionVariants } from '@/app/motion';

function Modal({ isOpen, onClose }) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            className="fixed inset-0 bg-black/60"
            variants={motionVariants.modal.overlay}
            initial="initial"
            animate="animate"
            exit="exit"
            onClick={onClose}
          />
          <motion.div
            className="fixed inset-0 flex items-center justify-center"
            variants={motionVariants.modal.content}
            initial="initial"
            animate="animate"
            exit="exit"
          >
            <div className="bg-gray-900 p-6 rounded-lg">
              Modal content
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
```

---

## 10. Ambient Background

```typescript
import { AmbientBackground } from '@/app/motion';

function App() {
  return (
    <div className="relative min-h-screen">
      <AmbientBackground
        activity="active"
        tradingActive={isTrading}
        marketVolatility={volatility}
      />

      {/* Your dashboard content */}
    </div>
  );
}
```

---

## Common Patterns

### Portfolio Card with Live Updates

```typescript
import {
  GlassCard,
  PriceFlash,
  StateBadge,
  motionTokens,
} from '@/app/motion';
import { AnimatedCurrency } from '@/app/components/animated/AnimatedNumber';

function PortfolioCard({ equity, prevEquity, riskState }) {
  return (
    <GlassCard hoverable variant="elevated">
      <div className="relative p-6">
        <PriceFlash value={equity} previousValue={prevEquity} />

        <div className="flex justify-between">
          <div>
            <h3 className="text-sm text-gray-400">Total Equity</h3>
            <AnimatedCurrency
              value={equity}
              className="text-3xl font-bold"
              flashOnChange
            />
          </div>

          <StateBadge
            state={riskState === 'NORMAL' ? 'active' : 'warning'}
            label={riskState}
          />
        </div>
      </div>
    </GlassCard>
  );
}
```

### Event Feed with Pipeline

```typescript
import { GlassPanel, PipelineFlow, EventCounter } from '@/app/motion';

function EventFeed({ events, signalCount, isActive }) {
  return (
    <GlassPanel borderColor={motionTokens.colors.normal} animate>
      <div className="p-4">
        <div className="flex justify-between mb-4">
          <h3>Live Events</h3>
          <EventCounter
            count={signalCount}
            label="Signals"
            active={isActive}
          />
        </div>

        <PipelineFlow events={events} maxVisible={5} />
      </div>
    </GlassPanel>
  );
}
```

### Control Panel

```typescript
import {
  GlassCard,
  AnimatedToggle,
  RippleButton,
  ConnectionStateIndicator,
} from '@/app/motion';

function ControlPanel({ wsStatus, tradingEnabled, onToggle, onStart }) {
  return (
    <GlassCard>
      <div className="p-6 space-y-4">
        <ConnectionStateIndicator state={wsStatus} showLabel />

        <AnimatedToggle
          enabled={tradingEnabled}
          onChange={onToggle}
          label="Enable Auto-Trading"
        />

        <RippleButton
          variant="primary"
          onClick={onStart}
          disabled={!tradingEnabled}
        >
          Start Trading
        </RippleButton>
      </div>
    </GlassCard>
  );
}
```

---

## Motion Tokens Reference

```typescript
import { motionTokens } from '@/app/motion';

// Durations
motionTokens.duration.instant   // 50ms
motionTokens.duration.fast      // 150ms
motionTokens.duration.normal    // 200ms
motionTokens.duration.slow      // 400ms

// Easing
motionTokens.ease.snappy        // Default
motionTokens.ease.smooth        // Smooth in-out
motionTokens.ease.bounce        // Slight overshoot

// Colors
motionTokens.colors.connected   // #00FF88 (green)
motionTokens.colors.normal      // #00D9FF (cyan)
motionTokens.colors.softBrake   // #F59E0B (amber)
motionTokens.colors.hardStop    // #EF4444 (red)
motionTokens.colors.profit      // #00FF88
motionTokens.colors.loss        // #FF3366
```

---

## Performance Rules

### ✅ DO

- Use `transform` and `opacity` only
- Use `safeTransition()` for reduced motion support
- Keep animations under 400ms
- Remove `will-change` after animation

### ❌ DON'T

- Animate `width`, `height`, `top`, `left`
- Create infinite loops without state change
- Add motion that competes with price data
- Forget accessibility (reduced motion)

---

## Testing

```bash
# Build check
npm run build

# Run tests
npm run test

# Visual check
1. Open Chrome DevTools
2. Performance tab
3. Record interaction
4. Check for 60fps, no layout thrashing
```

---

## Next Steps

1. **Read full guide:** `MOTION_DESIGN_GUIDE.md`
2. **See visual demos:** `MOTION_VISUAL_DEMO.md`
3. **Check examples:** `apps/ui/src/app/motion/examples/`
4. **Integrate:** Start with simple components, add motion gradually

---

## Quick Troubleshooting

**Animations not playing?**
- Check browser console for errors
- Verify framer-motion is installed
- Ensure parent has `position: relative`

**Performance issues?**
- Use only `transform` and `opacity`
- Check for layout thrashing in DevTools
- Reduce concurrent animations

**Reduced motion not working?**
- Use `safeTransition()` instead of raw duration
- Test with `prefers-reduced-motion: reduce` in DevTools

---

## Support

See full documentation in:
- `MOTION_DESIGN_GUIDE.md` - Complete reference
- `MOTION_VISUAL_DEMO.md` - Visual examples
- `apps/ui/src/app/motion/` - Source code

---

**Ready to animate!** 🎬
