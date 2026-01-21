'use client';

export default function ReconnectingOverlay() {
  return (
    <div className="reconnecting-overlay">
      <div className="hean-card max-w-md text-center">
        <div className="animate-pulse mb-4">
          <div className="text-4xl mb-4">⚡</div>
          <h2 className="text-2xl font-bold hean-text-glow mb-2">
            Reconnecting...
          </h2>
          <p className="text-hean-secondary">
            Attempting to reconnect to HEAN API
          </p>
          <div className="mt-4 flex justify-center">
            <div className="w-8 h-8 border-4 border-hean-primary border-t-transparent rounded-full animate-spin"></div>
          </div>
        </div>
      </div>
    </div>
  );
}