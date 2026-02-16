"use client";

import dynamic from "next/dynamic";

// Disable SSR for 3D components
const InteractiveTour = dynamic(() => import("@/sections/InteractiveTour"), {
  ssr: false,
});

export default function ClientTour() {
  return <InteractiveTour />;
}
