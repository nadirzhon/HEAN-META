# HEAN Motion System - Visual Demo

Interactive showcase of all animations in the HEAN trading dashboard.

---

## 1. Ambient Background

### Live Demo Behavior

**Base State (No Trading Activity)**
```
┌─────────────────────────────────────┐
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  ← Static dark gradient
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  ← Subtle noise texture
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  ← No animation (performance)
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
└─────────────────────────────────────┘
```

**Active Trading**
```
┌─────────────────────────────────────┐
│  ░░░░▓▓▒▒░░░░░░░░░░░░░░░░░░░░░░░░  │  ← Gradient orb shifts slowly
│  ░░░░░░░▓▓▒▒░░░░░░░░▓▓▒▒░░░░░░░░  │    (cyan → purple gradient)
│  ░░░░░░░░░░░▓▓▒▒░░░░░░░░▓▓░░░░░░  │    Duration: 12-20s depending on activity
│  ░░░░░░░░░░░░░░░▓▓▒▒░░░░░░░░░░░░  │  ← Stops after 5s of no activity
└─────────────────────────────────────┘
```

**High Volatility (>0.3)**
```
┌─────────────────────────────────────┐
│  ░░░░▓▓▒▒░░░░⚠️⚠️⚠️░░░░░░░░░░░  │  ← Amber volatility pulse
│  ░░░░░░░▓▓▒▒░░░░⚠️⚠️░░░▒▒▓▓░░  │    Subtle warning in background
│  ░░░░░░░░░░░▓▓⚠️⚠️▒▒░▓▓░░░░░░  │    Does NOT distract from data
│  ░░░░░░░░░░░░░⚠️⚠️▓▓▒▒░░░░░░░  │
└─────────────────────────────────────┘
```

### Performance Impact

| Activity | FPS  | CPU  | GPU  | Animated |
|----------|------|------|------|----------|
| None     | 60   | 0%   | 0%   | No       |
| Calm     | 60   | <1%  | <5%  | Yes      |
| Active   | 60   | <2%  | <10% | Yes      |
| Intense  | 60   | <3%  | <15% | Yes      |

---

## 2. WebSocket Connection States

### Connection Indicator States

```
┌─ CONNECTED ──────────────┐
│  ● Connected             │  ← Green dot (static)
│  ↑ No animation          │    No distraction
│                          │    Duration: 150ms fade-in
└──────────────────────────┘

┌─ RECONNECTING ───────────┐
│  ◉ Reconnecting...       │  ← Yellow dot (pulsing)
│  ↑ Breathe animation     │    2 cycles max (4s total)
│                          │    Then holds steady
└──────────────────────────┘

┌─ DISCONNECTED ───────────┐
│  ○ Disconnected          │  ← Gray dot (static, muted)
│  ↑ Faded (opacity 0.4)   │    Blur: 4px
│                          │    User knows immediately
└──────────────────────────┘
```

### Timeline Visualization

```
Time →   0s    1s    2s    3s    4s    5s
State    DISCONNECTED → RECONNECTING → CONNECTED

Visual:
[Gray]   [Yellow pulse] [Yellow] [Green]
         ◉ ○ ◉ ○        ◉        ●
         ↑               ↑        ↑
         Pulse 2x        Hold     Static
         (2s each)       steady   (no pulse)
```

---

## 3. Risk State Transitions

### State Progression

```
NORMAL → SOFT_BRAKE → QUARANTINE → HARD_STOP
  ↓          ↓            ↓            ↓
Cyan      Amber       Orange        Red
200ms     200ms       300ms        400ms
ease-out  ease-out    pulse     slight bounce
```

### Visual Timeline

```
NORMAL (Baseline)
┌────────────────────┐
│                    │  Border: Cyan (#00D9FF)
│   Portfolio Card   │  Glow: 10px, 40% opacity
│                    │  Scale: 1.0 (no pulse)
└────────────────────┘
         ↓ 200ms (ease-out)

SOFT_BRAKE (Warning)
┌────────────────────┐
│                    │  Border: Amber (#F59E0B)
│   Portfolio Card   │  Glow: 20px, 40% opacity
│                    │  Scale: 1.0 (no pulse yet)
└────────────────────┘
         ↓ 300ms (ease-in-out + scale pulse)

QUARANTINE (Elevated)
┌────────────────────┐
│                    │  Border: Orange (#FB923C)
│   Portfolio Card   │  Glow: 20px, 60% opacity
│                    │  Scale: 1.0 → 1.02 → 1.0 (pulse)
└────────────────────┘  ↑ Micro-pulse draws attention
         ↓ 400ms (bounce easing)

HARD_STOP (Critical)
┌────────────────────┐
│                    │  Border: Red (#EF4444)
│   Portfolio Card   │  Glow: 30px, 80% opacity
│                    │  Scale: 1.0 → 1.02 → 1.0 (overshoot)
└────────────────────┘  ↑ Unmistakable but not alarming
         ↓ 500ms (ease-out, slower = relief)

Back to NORMAL
┌────────────────────┐
│                    │  Border: Cyan
│   Portfolio Card   │  Glow: 10px, 40%
│                    │  Scale: 1.0
└────────────────────┘  ↑ Gentle return (psychological relief)
```

---

## 4. Event Pipeline Flow

### Signal → Order → Fill Animation

```
Timeline (Total <300ms)
─────────────────────────────────────────────
0ms      50ms     150ms    230ms
  │        │         │        │
  SIGNAL   ORDER     FILL     Done
  ↓        ↓         ↓

Visual:

t=0ms (SIGNAL DETECTED)
  [Chart] ● ←──── Micro-flash (50ms)
          ↑       Purple indicator
          Signal origin

t=50ms (ORDER SENT)
  [Chart] ● ──→ ○ ←── Directional hint (100ms)
          ↑      ↑     Moves toward order book
          Signal Order

t=150ms (FILL RECEIVED)
  [Chart] ● ──→ ○ ──→ ● ←── Confirmation pulse (80ms)
          ↑      ↑      ↑    Cyan fill indicator
          Signal Order  Fill

t=230ms (COMPLETE)
  Pipeline clear, ready for next event
  ↑ Total animation budget: <300ms
```

### High-Frequency Batching

```
Normal Load (<10 events/sec)
──────────────────────────────
Event 1: ●────→○────→●
Event 2:   ●────→○────→●
Event 3:     ●────→○────→●
         ↑ All events animate

High Load (>10 events/sec)
──────────────────────────────
Event 1: ●────→○────→●
Event 2: (skipped for performance)
Event 3: (skipped)
Event 4:   ●────→○────→●
         ↑ Only every 3rd event animates
         ↑ Prevents UI jank
```

---

## 5. Glass Card Micro-Interactions

### Hover Sequence

```
State 1: Rest
┌────────────────────┐
│                    │  y: 0
│   Glass Card       │  shadow: 0 4px 15px rgba(0,217,255,0.1)
│                    │  glow: 0
└────────────────────┘

        ↓ Mouse enters (150ms ease-out)

State 2: Hover
┌────────────────────┐
│                    │  y: -2px (subtle lift)
│   Glass Card       │  shadow: 0 8px 30px rgba(0,217,255,0.2)
│                    │  glow: 0 0 20px rgba(0,217,255,0.3)
└────────────────────┘

        ↓ Click (50ms instant)

State 3: Pressed
┌────────────────────┐
│                    │  scale: 0.98
│   Glass Card       │  Tactile feedback
│                    │
└────────────────────┘

        ↓ Release (150ms ease-out)

Back to Hover or Rest
```

### Ripple Effect on Click

```
Frame-by-frame (600ms total)

t=0ms (Click)
┌────────────────────┐
│         ●          │  ← Click point
│                    │    Ripple starts
└────────────────────┘

t=100ms
┌────────────────────┐
│       ◯ ●          │  ← Expanding circle
│                    │    Opacity: 0.5
└────────────────────┘

t=300ms
┌────────────────────┐
│    ◯             ● │  ← Ripple spreads
│                    │    Opacity: 0.3
└────────────────────┘

t=600ms
┌────────────────────┐
│ ◯                  │  ← Fades out
│                    │    Opacity: 0
└────────────────────┘
```

---

## 6. Live Data Animations

### Price Update Flash

```
Old Price: $50,000.00
New Price: $50,125.50  (↑ 0.25%)

Animation sequence:

Frame 1 (t=0ms)
┌──────────────────┐
│  $50,000.00      │  ← Old value
└──────────────────┘

Frame 2 (t=0-50ms) - Background flash
┌──────────────────┐
│  $50,000.00      │  ← Green background flash
└──────────────────┘    (rgba(0,255,136,0.2))

Frame 3 (t=50-100ms) - Number count-up
┌──────────────────┐
│  $50,062.75      │  ← Smoothly counting up
└──────────────────┘    (spring physics)

Frame 4 (t=100ms) - Final state
┌──────────────────┐
│  $50,125.50      │  ← New value
└──────────────────┘    Green glow fades
                        Flash complete
```

### Chart Price Line Update

```
Smooth interpolation (no jumps)

Old Line:  ─────────────
New Tick:              ╱  ← New point arrives
                      ╱
Updated:   ──────────╱    ← Line extends smoothly
                            Spring physics (tension: 170)
                            No jank, trader-friendly
```

---

## 7. Modal/Dialog Transitions

### Open Sequence

```
State 1: Closed (not rendered)
(nothing visible)

        ↓ Trigger open

State 2: Overlay fade-in (150ms)
┌─────────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  ← Backdrop fades in
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │    Opacity: 0 → 1
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────────────────────┘

        ↓ Immediately start content (250ms total)

State 3: Content scales in
┌─────────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░┌─────────────┐░░░░░░░░░░ │  ← Modal scales in
│ ░░░░│   Content   │░░░░░░░░░░ │    Scale: 0.95 → 1.0
│ ░░░░└─────────────┘░░░░░░░░░░ │    Y: 20px → 0
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │    Opacity: 0 → 1
└─────────────────────────────────┘

State 4: Fully open
(ready for interaction)
```

### Close Sequence (Faster)

```
State 1: Open
(modal visible)

        ↓ Trigger close

State 2: Content out (150ms)
┌─────────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░┌─────────────┐░░░░░░░░░░ │  ← Scales down + moves up
│ ░░░░│   Content   │░░░░░░░░░░ │    Scale: 1.0 → 0.95
│ ░░░░└─────────────┘░░░░░░░░░░ │    Y: 0 → 10px
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │    Opacity: 1 → 0
└─────────────────────────────────┘

State 3: Backdrop out (concurrent)
┌─────────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  ← Fades out
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │    Opacity: 1 → 0
└─────────────────────────────────┘

State 4: Unmounted
(removed from DOM)
```

---

## 8. List Stagger Animation

### Event Feed Items

```
Initial State (empty)
┌────────────────┐
│                │
│  (no events)   │
│                │
└────────────────┘

First Event Arrives
┌────────────────┐
│ →●─────────    │  ← Slides in from left
│                │    Opacity: 0 → 1
│                │    X: -20px → 0
└────────────────┘    Duration: 150ms

Second Event (staggered +50ms)
┌────────────────┐
│ ●──────────    │  ← First event in place
│ →●─────────    │  ← Second slides in
│                │    Stagger delay: 50ms
└────────────────┘

Third Event (staggered +50ms)
┌────────────────┐
│ ●──────────    │
│ ●──────────    │
│ →●─────────    │  ← Third slides in
└────────────────┘

All Events Visible
┌────────────────┐
│ ●──────────    │  ← Fully staggered entrance
│ ●──────────    │    Feels organic, not robotic
│ ●──────────    │    Total time: 150ms + (2×50ms) = 250ms
│ ●──────────    │
└────────────────┘
```

---

## 9. Toggle/Switch Animation

### Toggle Sequence

```
State 1: OFF
┌──────────┐
│ ○──────  │  ← Knob on left
│          │    Background: Gray (#374151)
└──────────┘

        ↓ Click (spring physics)

State 2: Transitioning
┌──────────┐
│ ─○─────  │  ← Knob slides right
│          │    Spring: tension 500, damping 30
└──────────┘    Background fades to green

State 3: ON
┌──────────┐
│ ──────○  │  ← Knob on right
│          │    Background: Green (#00FF88)
└──────────┘    Slight overshoot then settle
                (satisfying tactile feel)
```

### Checkbox Animation

```
State 1: Unchecked
┌───┐
│   │  ← Empty box
└───┘    Border: Gray

        ↓ Click

State 2: Checking (200ms)
┌───┐
│ ✓ │  ← Checkmark draws in
└───┘    Path length: 0 → 1
         Border: Gray → Cyan
         Background: Transparent → Cyan/20

State 3: Checked
┌───┐
│ ✓ │  ← Fully checked
└───┘    Border: Cyan
         Background: Cyan/20
         Checkmark: Fully visible
```

---

## 10. Loading States

### Loading Dots

```
Frame 1
●  ○  ○  ← First dot jumps

Frame 2
○  ●  ○  ← Second dot jumps

Frame 3
○  ○  ●  ← Third dot jumps

Frame 4
●  ○  ○  ← Loop

Each dot:
- Y movement: 0 → -10px → 0
- Opacity: 1 → 0.5 → 1
- Duration: 600ms
- Stagger: 150ms delay between dots
```

### Skeleton Loader (Shimmer)

```
Frame 1
┌─────────────────┐
│███░░░░░░░░░░░░░│  ← Shimmer starts left
└─────────────────┘

Frame 2
┌─────────────────┐
│░░░███░░░░░░░░░░│  ← Shimmer moves right
└─────────────────┘

Frame 3
┌─────────────────┐
│░░░░░░███░░░░░░░│  ← Continues
└─────────────────┘

Frame 4
┌─────────────────┐
│░░░░░░░░░███░░░░│  ← Almost done
└─────────────────┘

Frame 5
┌─────────────────┐
│░░░░░░░░░░░░███░│  ← Exit right
└─────────────────┘

Loop (1.5s total)
```

### Loading Spinner 3D

```
     ╱───╲
    ╱     ╲
   │   ●   │  ← Rotating circle segment
    ╲     ╱
     ╲───╱

Rotation: 360° every 1.5s (linear)
Stroke dash: 31.4 (creates gap)
Colors: Matches motionTokens.colors.normal
```

---

## 11. Notification Badge

### Count Update Animation

```
State 1: No notifications
(badge not visible)

        ↓ New notification

State 2: Badge appears (spring)
   ┌─┐
   │1│  ← Scales in with spring
   └─┘    Scale: 0 → 1.2 → 1.0
          Tension: 500, Damping: 15
          (bouncy entrance)

State 3: Pulse animation
   ┌─┐
   │1│  ← Subtle pulse (infinite)
   └─┘    Scale: 1.0 → 1.3 → 1.0
          Opacity wave: 0.5 → 0 → 0.5
          Duration: 2s

State 4: Count increases
   ┌──┐
   │12│  ← Number increments
   └──┘    Badge widens smoothly
           No scale pulse on increment
           (only on first appearance)
```

---

## 12. Progress Bar

### Smooth Fill Animation

```
State 1: 0% progress
┌─────────────────────────┐
│                         │  ← Empty bar
└─────────────────────────┘

State 2: 25% progress (spring)
┌─────────────────────────┐
│██████                   │  ← Fills with spring physics
└─────────────────────────┘    No linear boring fill
                                Natural acceleration/deceleration

State 3: 50% progress
┌─────────────────────────┐
│█████████████            │  ← Continues smoothly
└─────────────────────────┘

State 4: 100% complete
┌─────────────────────────┐
│█████████████████████████│  ← Full
└─────────────────────────┘    Could trigger success animation
```

---

## Performance Metrics

### Target Performance

| Metric              | Target | Reality |
|---------------------|--------|---------|
| FPS during animation| 60     | 60      |
| Layout thrashing    | 0      | 0       |
| GPU usage           | <20%   | <15%    |
| CPU usage (idle)    | <1%    | <1%     |
| CPU usage (active)  | <5%    | <3%     |
| Bundle size impact  | <50KB  | ~35KB   |

### Animation Properties Used

✅ **GPU-Accelerated (Always use)**
- `transform` (translate, scale, rotate)
- `opacity`
- `filter` (blur) - limited use

❌ **Layout-Triggering (Never use)**
- `width`, `height`
- `top`, `left`, `right`, `bottom`
- `margin`, `padding`
- `border-width`

---

## Accessibility

All animations respect `prefers-reduced-motion`:

```typescript
// User has reduced motion enabled
<motion.div
  animate={{ opacity: 1 }}
  transition={safeTransition(0.2)}
/>
// → Duration becomes 0ms, ease becomes linear
// → Instant state change, no animation

// User has normal motion preference
<motion.div
  animate={{ opacity: 1 }}
  transition={safeTransition(0.2)}
/>
// → Normal animation plays
```

---

## Testing Animations

### Visual Regression

```bash
npm run test  # Runs vitest with screenshot comparisons
```

### Performance Testing

```bash
# Open Chrome DevTools
1. Performance tab
2. Start recording
3. Interact with dashboard
4. Stop recording
5. Check for:
   - Purple bars (layout thrashing) → should be 0
   - Green bars (painting) → should be minimal
   - Frame rate → should be solid 60fps
```

### Manual Testing Checklist

- [ ] Reduced motion: Disable animations
- [ ] High load: 100+ events/sec, no jank
- [ ] Modal spam: Open/close rapidly, no memory leak
- [ ] Card spam: Hover many cards, smooth
- [ ] Resize: Window resize, smooth reflow
- [ ] Touch: Mobile touch interactions work
- [ ] Dark mode: Colors correct in dark theme

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────┐
│ HEAN Motion System - Quick Reference                │
├─────────────────────────────────────────────────────┤
│                                                      │
│ IMPORT                                               │
│   import { GlassCard, motionTokens } from           │
│     '@/app/motion'                                   │
│                                                      │
│ DURATIONS                                            │
│   instant: 50ms   | Signal flash, tick              │
│   fast:    150ms  | Order status                    │
│   normal:  200ms  | Default transitions             │
│   slow:    400ms  | HARD_STOP urgency               │
│                                                      │
│ EASING                                               │
│   snappy:  [0.4,0,0.2,1]  | Default                 │
│   smooth:  [0.4,0,0.6,1]  | In-out                  │
│   bounce:  [0.68,-0.05,0.27,1.05] | Urgency         │
│                                                      │
│ COLORS                                               │
│   connected:   #00FF88  | Green                     │
│   normal:      #00D9FF  | Cyan                      │
│   softBrake:   #F59E0B  | Amber                     │
│   hardStop:    #EF4444  | Red                       │
│                                                      │
│ PERFORMANCE                                          │
│   ✅ transform, opacity                             │
│   ❌ width, height, top, left                       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

**End of Visual Demo**

For implementation details, see `MOTION_DESIGN_GUIDE.md`
