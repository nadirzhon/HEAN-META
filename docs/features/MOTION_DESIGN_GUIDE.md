# HEAN Motion Design Guide

Complete animation system for the HEAN trading dashboard. Every motion serves a clear purpose: communicating system state to traders without distraction.

## Philosophy

**Purpose-Driven Motion**: Every animation must answer "What state change am I communicating?"

**Performance First**: All animations GPU-accelerated (transform + opacity only), <400ms duration for critical updates.

**Trader-Focused**: Motion guides attention to what matters, never competes with price action or critical data.

---

## Quick Start

### Installation

All dependencies already installed. Import from centralized module:

```typescript
import {
  motionTokens,
  GlassCard,
  ConnectionStateIndicator,
  EventFlash,
} from '@/app/motion';
```

### Basic Example

```typescript
import { GlassCard, ConnectionStateIndicator } from '@/app/motion';

function DashboardPanel() {
  return (
    <GlassCard hoverable>
      <div className="p-4">
        <ConnectionStateIndicator state="connected" />
        <h2>Portfolio</h2>
        {/* ... */}
      </div>
    </GlassCard>
  );
}
```

---

## Motion Tokens

Centralized timing, easing, and color constants.

### Duration

```typescript
motionTokens.duration.instant   // 50ms  - Signal flash, tick update
motionTokens.duration.micro     // 80ms  - Fill confirmation
motionTokens.duration.flash     // 100ms - Price change
motionTokens.duration.fast      // 150ms - Order status update
motionTokens.duration.normal    // 200ms - Default transitions
motionTokens.duration.snappy    // 250ms - Modal open
motionTokens.duration.moderate  // 300ms - SOFT_BRAKE→QUARANTINE
motionTokens.duration.slow      // 400ms - QUARANTINE→HARD_STOP
motionTokens.duration.relief    // 500ms - Any state→NORMAL
```

### Easing

```typescript
motionTokens.ease.snappy      // [0.4, 0, 0.2, 1] - Default (Tailwind ease-out)
motionTokens.ease.smooth      // [0.4, 0, 0.6, 1] - Smooth in-out
motionTokens.ease.bounce      // Slight overshoot (for HARD_STOP urgency)
```

### Colors

```typescript
motionTokens.colors.connected     // #00FF88 - WebSocket connected
motionTokens.colors.reconnecting  // #FFCC00 - Reconnecting pulse
motionTokens.colors.normal        // #00D9FF - Normal risk state
motionTokens.colors.softBrake     // #F59E0B - Soft brake warning
motionTokens.colors.quarantine    // #FB923C - Quarantine state
motionTokens.colors.hardStop      // #EF4444 - Hard stop critical
motionTokens.colors.profit        // #00FF88 - Profit flash
motionTokens.colors.loss          // #FF3366 - Loss flash
```

---

## Component Reference

### 1. State Transitions

#### ConnectionStateIndicator

Shows WebSocket connection status with appropriate animation.

```typescript
<ConnectionStateIndicator state="connected" showLabel />
```

**Props:**
- `state`: `'connected' | 'reconnecting' | 'disconnected'`
- `showLabel`: boolean (optional)

**Behavior:**
- `connected`: Green dot, no pulse
- `reconnecting`: Yellow dot, breathe animation (max 2 cycles)
- `disconnected`: Gray dot, faded

#### RiskStateBorder

Communicates risk level through border color and glow.

```typescript
<RiskStateBorder state="softBrake">
  <div className="p-4">Risk content</div>
</RiskStateBorder>
```

**Props:**
- `state`: `'normal' | 'softBrake' | 'quarantine' | 'hardStop'`

**Timing:**
- NORMAL → SOFT_BRAKE: 200ms, ease-out
- SOFT_BRAKE → QUARANTINE: 300ms, ease-in-out, scale pulse
- QUARANTINE → HARD_STOP: 400ms, bounce easing
- Any state → NORMAL: 500ms (slower for psychological relief)

#### StateBadge

Displays current system state with appropriate color and pulse.

```typescript
<StateBadge state="warning" label="High Volatility" />
```

**Props:**
- `state`: `'active' | 'warning' | 'critical' | 'neutral'`
- `label`: string
- `variant`: `'default' | 'minimal'`

---

### 2. Event Flow

#### EventFlash

Micro-flash animation at event origin (Signal/Order/Fill).

```typescript
<EventFlash type="signal" side="buy" />
```

**Duration:** 50ms (instant)
**Usage:** Overlay on chart markers, order buttons

#### PipelineFlow

Visualizes Signal → Order → Fill pipeline.

```typescript
const [events, setEvents] = useState<TradingEvent[]>([]);

<PipelineFlow events={events} maxVisible={5} />
```

**Performance:**
- Auto-batches under high load (>10 events/sec)
- Auto-removes events after 2s
- Total animation budget: <300ms per event

#### EventCounter

Live counter with pulse on activity.

```typescript
<EventCounter count={signalCount} label="Signals" active={isTrading} />
```

#### PriceFlash

Brief background flash on price change.

```typescript
<div className="relative">
  <PriceFlash value={currentPrice} previousValue={prevPrice} />
  <span>{currentPrice}</span>
</div>
```

---

### 3. Glass Cards

#### GlassCard

Interactive glassmorphic card with hover effects.

```typescript
<GlassCard
  hoverable
  clickable
  variant="elevated"
  glowColor={motionTokens.colors.normal}
  parallax
  onClick={handleClick}
>
  <div className="p-6">Card content</div>
</GlassCard>
```

**Props:**
- `hoverable`: Enable hover lift effect
- `clickable`: Enable press feedback
- `variant`: `'default' | 'elevated' | 'minimal'`
- `parallax`: Enable mouse-follow 3D effect
- `glowColor`: Custom glow color

**Performance:**
- Only animates `transform` and `opacity`
- Uses `will-change` during interaction
- GPU-accelerated

#### ExpandableGlassSection

Accordion-style expandable panel.

```typescript
<ExpandableGlassSection
  title="Advanced Settings"
  icon={<SettingsIcon />}
  defaultExpanded={false}
>
  <div>Content here</div>
</ExpandableGlassSection>
```

#### DraggableGlassCard

Rearrangeable card for dashboard customization.

```typescript
<DraggableGlassCard onDragEnd={(info) => console.log(info.offset)}>
  <div>Draggable content</div>
</DraggableGlassCard>
```

---

### 4. Ambient Background

#### AmbientBackground

Composite ambient effects (gradient + grid + scanline + cursor glow).

```typescript
<AmbientBackground
  activity="active"           // 'calm' | 'active' | 'intense'
  marketVolatility={0.7}      // 0-1
  tradingActive={isTrading}
  showGrid
  showScanline={false}
  showCursorGlow
/>
```

**Behavior:**
- Stops animating after 5s of inactivity (performance)
- Gradient shifts based on activity level
- Volatility pulse appears when volatility > 0.3

#### Individual Components

```typescript
<AmbientGradient activity="active" marketVolatility={0.5} tradingActive />
<AmbientGrid active opacity={0.05} />
<ScanlineEffect active />
<CursorGlow enabled />
```

---

### 5. 3D Elements

#### ThreeDIndicator

Rotating 3D orb for system state.

```typescript
<ThreeDIndicator state="active" size={100} />
```

**States:**
- `active`: Green, pulsing
- `warning`: Yellow, pulsing
- `critical`: Red, strong pulse
- `neutral`: Gray, static

#### DepthCard

Card with mouse-follow 3D tilt.

```typescript
<DepthCard>
  <div className="p-4">Content with depth</div>
</DepthCard>
```

**Max rotation:** 5deg

#### ParallaxLayer

Layer with depth-based scroll parallax.

```typescript
<ParallaxLayer depth={0.5}>
  <div>Background element</div>
</ParallaxLayer>
```

**Depth:**
- `0`: Background (moves slower)
- `1`: Foreground (moves with scroll)

---

### 6. Micro-Interactions

#### RippleButton

Button with click ripple effect.

```typescript
<RippleButton
  variant="primary"
  onClick={handleClick}
  disabled={false}
>
  Execute Trade
</RippleButton>
```

**Variants:**
- `default`: White/10 background
- `primary`: Cyan accent
- `danger`: Red accent

#### AnimatedToggle

Smooth toggle switch.

```typescript
<AnimatedToggle
  enabled={tradingEnabled}
  onChange={setTradingEnabled}
  label="Enable Trading"
  size="md"
/>
```

#### AnimatedCheckbox

Checkbox with checkmark draw animation.

```typescript
<AnimatedCheckbox
  checked={agreed}
  onChange={setAgreed}
  label="I understand the risks"
/>
```

#### SuccessCheckmark

Animated success indicator.

```typescript
<SuccessCheckmark size={60} color={motionTokens.colors.connected} />
```

**Animation:**
- Circle draws in (500ms)
- Checkmark draws in (400ms, delayed)
- Scales from 0 with rotation

#### LoadingDots

Three-dot loading indicator.

```typescript
<LoadingDots size={8} color={motionTokens.colors.normal} />
```

#### SkeletonLoader

Shimmer loading placeholder.

```typescript
<SkeletonLoader width="100%" height={20} />
<SkeletonLoader width={200} height={40} className="mt-2" />
```

#### NotificationBadge

Pulsing count badge.

```typescript
<div className="relative">
  <BellIcon />
  <NotificationBadge count={unreadCount} max={99} />
</div>
```

#### ProgressBar

Animated progress indicator.

```typescript
<ProgressBar
  progress={uploadProgress}
  color={motionTokens.colors.connected}
  height={4}
  showLabel
/>
```

---

## Performance Checklist

### Before Deployment

- [ ] Only `transform` and `opacity` animated
- [ ] `will-change` used sparingly, removed after animation
- [ ] No forced synchronous layouts
- [ ] `AnimatePresence` used for exit animations
- [ ] Reduced motion media query respected (`safeTransition`)
- [ ] 60fps maintained during animation (Chrome DevTools)
- [ ] No console warnings about layout thrashing

### Testing

```bash
# Run build
npm run build

# Check bundle size
# Motion system should add <50KB gzipped

# Lighthouse performance score
# Should maintain 90+ score

# Visual regression
npm run test
```

---

## Integration Examples

### Portfolio Card with Live Updates

```typescript
import { GlassCard, PriceFlash, StateBadge } from '@/app/motion';

function PortfolioCard({ equity, prevEquity, riskState }) {
  return (
    <GlassCard hoverable variant="elevated">
      <div className="relative p-6">
        <PriceFlash value={equity} previousValue={prevEquity} />

        <div className="flex justify-between items-start">
          <div>
            <h3 className="text-sm text-gray-400">Total Equity</h3>
            <p className="text-3xl font-bold">${equity.toFixed(2)}</p>
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
import { PipelineFlow, EventFlash, GlassPanel } from '@/app/motion';

function EventFeed({ events }) {
  return (
    <GlassPanel borderColor={motionTokens.colors.normal} animate>
      <div className="p-4">
        <h3 className="text-lg font-semibold mb-4">Live Events</h3>

        <PipelineFlow events={events} maxVisible={10} />

        <div className="mt-4 space-y-2">
          {events.map(event => (
            <div key={event.id} className="relative p-2 bg-white/5 rounded">
              <EventFlash type={event.type} side={event.side} />
              <span>{event.symbol} {event.type}</span>
            </div>
          ))}
        </div>
      </div>
    </GlassPanel>
  );
}
```

### Control Panel with Toggles

```typescript
import {
  GlassCard,
  AnimatedToggle,
  RippleButton,
  ConnectionStateIndicator,
} from '@/app/motion';

function ControlPanel({ wsState, tradingEnabled, onToggle, onStart }) {
  return (
    <GlassCard>
      <div className="p-6 space-y-4">
        <ConnectionStateIndicator state={wsState} showLabel />

        <AnimatedToggle
          enabled={tradingEnabled}
          onChange={onToggle}
          label="Enable Auto-Trading"
          size="lg"
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

### Full Dashboard Layout

```typescript
import {
  AmbientBackground,
  GlassCard,
  DepthCard,
  ParallaxLayer,
} from '@/app/motion';

function TradingDashboard() {
  return (
    <div className="relative min-h-screen">
      <AmbientBackground
        activity="active"
        tradingActive={isTrading}
        marketVolatility={volatility}
      />

      <ParallaxLayer depth={0.3}>
        <div className="grid grid-cols-3 gap-4 p-8">
          <DepthCard>
            <GlassCard variant="elevated">
              {/* Portfolio content */}
            </GlassCard>
          </DepthCard>

          {/* More cards */}
        </div>
      </ParallaxLayer>
    </div>
  );
}
```

---

## Motion Patterns by Use Case

### WebSocket Connection States

```typescript
// CONNECTED → DISCONNECTED
<ConnectionStateIndicator state="disconnected" />
// - Fade to muted (0.4 opacity)
// - No pulse (static)

// DISCONNECTED → RECONNECTING
<ConnectionStateIndicator state="reconnecting" />
// - Gentle breathe animation (2 cycles max)
// - Yellow color

// RECONNECTING → CONNECTED
<ConnectionStateIndicator state="connected" />
// - Quick fade to normal (150ms)
// - Green color, static
```

### Risk State Transitions

```typescript
// NORMAL → SOFT_BRAKE
<RiskStateBorder state="softBrake">
  {/* 200ms, subtle border intensification */}
</RiskStateBorder>

// SOFT_BRAKE → QUARANTINE
<RiskStateBorder state="quarantine">
  {/* 300ms, scale pulse (1.0 → 1.02 → 1.0) */}
</RiskStateBorder>

// QUARANTINE → HARD_STOP
<RiskStateBorder state="hardStop">
  {/* 400ms, bounce easing, unmistakable */}
</RiskStateBorder>

// Any → NORMAL
<RiskStateBorder state="normal">
  {/* 500ms, gentle relief transition */}
</RiskStateBorder>
```

### Event Pipeline

```typescript
// Signal detected
<EventFlash type="signal" />  // 50ms micro-flash

// Order sent
<EventFlash type="order" side="buy" />  // 100ms directional hint

// Fill received
<EventFlash type="fill" />  // 80ms confirmation pulse

// Full pipeline
<PipelineFlow events={signalOrderFillChain} />
// Total: <300ms for complete chain
```

---

## Accessibility

All components respect `prefers-reduced-motion`:

```typescript
import { safeTransition } from '@/app/motion';

// Automatically disables animation if user prefers reduced motion
<motion.div
  animate={{ opacity: 1 }}
  transition={safeTransition(motionTokens.duration.normal)}
/>
```

---

## Troubleshooting

### Animations not playing

1. Check browser DevTools for layout thrashing warnings
2. Verify `framer-motion` version: `npm list framer-motion`
3. Ensure parent has `position: relative` for absolute positioned effects

### Performance issues

1. Use Chrome DevTools Performance tab
2. Check for forced reflows (look for purple bars)
3. Verify only `transform` and `opacity` are animating
4. Reduce number of concurrent animations

### Jank during animations

1. Add `will-change: transform` temporarily
2. Check for large DOM trees (>1000 nodes)
3. Consider using `useReducedMotion` hook for heavy scenes

---

## Contributing

When adding new animations:

1. **Define the purpose**: What state change does this communicate?
2. **Set timing**: Use `motionTokens.duration.*`
3. **Choose easing**: Use `motionTokens.ease.*`
4. **Test performance**: Maintain 60fps
5. **Test reduced motion**: Use `safeTransition()`
6. **Document**: Add to this guide with examples

---

## License

Part of HEAN trading system. Internal use only.
