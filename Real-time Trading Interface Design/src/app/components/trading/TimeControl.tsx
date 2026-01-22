import { useState } from "react";
import { Play, Pause, RotateCcw } from "lucide-react";
import { Card } from "@/app/components/ui/card";
import { Slider } from "@/app/components/ui/slider";
import { Button } from "@/app/components/ui/button";

interface TimeControlProps {
  onTimeChange?: (timestamp: number) => void;
  minTime: number;
  maxTime: number;
  currentTime: number;
}

export function TimeControl({ onTimeChange, minTime, maxTime, currentTime }: TimeControlProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [compressionLevel, setCompressionLevel] = useState(1); // 1x, 2x, 5x, 10x

  const totalDuration = maxTime - minTime;
  const currentPosition = ((currentTime - minTime) / totalDuration) * 100;

  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  };

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <Card className="p-4 bg-card/30 backdrop-blur-sm border-border/50">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-xs uppercase tracking-wider text-muted-foreground mb-1">
              Time Control
            </h4>
            <p className="text-xs text-muted-foreground/70">
              Replay system thinking
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-8 w-8 p-0"
              onClick={() => onTimeChange?.(minTime)}
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-8 w-8 p-0"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        {/* Timeline Slider */}
        <div className="space-y-2">
          <Slider
            value={[currentPosition]}
            onValueChange={([value]) => {
              const newTime = minTime + (value / 100) * totalDuration;
              onTimeChange?.(newTime);
            }}
            min={0}
            max={100}
            step={0.1}
            className="[&_[role=slider]]:bg-[var(--trading-cyan)] [&_[role=slider]]:border-[var(--trading-cyan)]"
          />
          <div className="flex items-center justify-between text-xs text-muted-foreground font-mono">
            <span>{formatTime(minTime)}</span>
            <span className="text-foreground">{formatTime(currentTime)}</span>
            <span>{formatTime(maxTime)}</span>
          </div>
        </div>

        {/* Compression Control */}
        <div className="pt-4 border-t border-border/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-muted-foreground uppercase tracking-wider">
              Time Compression
            </span>
            <span className="text-sm font-mono text-[var(--trading-cyan)]">
              {compressionLevel}x
            </span>
          </div>
          <Slider
            value={[compressionLevel]}
            onValueChange={([value]) => setCompressionLevel(value)}
            min={1}
            max={10}
            step={1}
            className="[&_[role=slider]]:bg-[var(--trading-amber)] [&_[role=slider]]:border-[var(--trading-amber)]"
          />
          <div className="flex justify-between text-xs text-muted-foreground mt-2">
            <span>Real-time</span>
            <span>10x compressed</span>
          </div>
          <p className="text-xs text-muted-foreground/70 mt-2">
            {formatDuration(totalDuration / compressionLevel)} of decision history
          </p>
        </div>

        {/* Playback Status */}
        {isPlaying && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <div className="h-2 w-2 rounded-full bg-[var(--trading-green)] animate-pulse" />
            <span>Replaying at {compressionLevel}x speed</span>
          </div>
        )}
      </div>
    </Card>
  );
}