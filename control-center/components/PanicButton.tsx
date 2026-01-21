'use client';

import { useState } from 'react';

export default function PanicButton() {
  const [isTriggering, setIsTriggering] = useState(false);
  const [lastTriggered, setLastTriggered] = useState<Date | null>(null);

  const handlePanic = async () => {
    if (!confirm('⚠️ EMERGENCY KILLSWITCH ⚠️\n\nThis will immediately halt all trading. Continue?')) {
      return;
    }

    setIsTriggering(true);

    try {
      const startTime = performance.now();
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/emergency/killswitch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const endTime = performance.now();
      const latency = endTime - startTime;

      if (!response.ok) {
        throw new Error('Failed to trigger killswitch');
      }

      const result = await response.json();
      setLastTriggered(new Date());
      
      console.log(`Killswitch triggered in ${latency.toFixed(2)}ms`);
      alert(`✅ Killswitch activated!\nResponse time: ${latency.toFixed(2)}ms\nReason: ${result.reason || 'Emergency stop'}`);
    } catch (error) {
      console.error('Error triggering killswitch:', error);
      alert('❌ Failed to trigger killswitch. Check console for details.');
    } finally {
      setIsTriggering(false);
    }
  };

  return (
    <div className="flex flex-col items-end gap-2">
      <button
        onClick={handlePanic}
        disabled={isTriggering}
        className={`
          hean-button-danger
          px-8 py-4 text-xl font-bold
          ${isTriggering ? 'opacity-50 cursor-not-allowed' : ''}
          animate-pulse-slow
        `}
      >
        {isTriggering ? 'TRIGGERING...' : '🚨 PANIC BUTTON 🚨'}
      </button>
      {lastTriggered && (
        <p className="text-xs text-hean-secondary">
          Last triggered: {lastTriggered.toLocaleTimeString()}
        </p>
      )}
    </div>
  );
}