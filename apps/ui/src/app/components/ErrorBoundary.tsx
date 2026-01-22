import React, { Component, ErrorInfo, ReactNode } from "react";
import { Card } from "@/app/components/ui/card";
import { Button } from "@/app/components/ui/button";
import { AlertTriangle } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-background flex items-center justify-center p-4">
          <Card className="max-w-2xl w-full p-6 border-[var(--trading-red)]/50 bg-[var(--trading-red)]/10">
            <div className="flex items-start gap-4">
              <AlertTriangle className="h-6 w-6 text-[var(--trading-red)] flex-shrink-0 mt-1" />
              <div className="flex-1 space-y-4">
                <div>
                  <h2 className="text-xl font-semibold text-[var(--trading-red)] mb-2">
                    UI Error Detected
                  </h2>
                  <p className="text-sm text-muted-foreground">
                    The UI encountered an unexpected error. This may have been caused by a render issue,
                    unhandled promise rejection, or state corruption.
                  </p>
                </div>

                {this.state.error && (
                  <div className="space-y-2">
                    <div className="text-sm font-semibold">Error:</div>
                    <div className="text-xs font-mono bg-background/50 p-3 rounded border border-border/50 overflow-auto max-h-40">
                      {this.state.error.toString()}
                    </div>
                  </div>
                )}

                {this.state.errorInfo && (
                  <div className="space-y-2">
                    <div className="text-sm font-semibold">Component Stack:</div>
                    <div className="text-xs font-mono bg-background/50 p-3 rounded border border-border/50 overflow-auto max-h-40">
                      {this.state.errorInfo.componentStack}
                    </div>
                  </div>
                )}

                <div className="flex gap-2 pt-2">
                  <Button onClick={this.handleReload} variant="default" className="flex-1">
                    Reload UI
                  </Button>
                  <Button onClick={this.handleReset} variant="outline" className="flex-1">
                    Try Again
                  </Button>
                </div>
              </div>
            </div>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}
