import type { Metadata } from "next";
import { Manrope, Inter } from "next/font/google";
import { ExplainabilityProvider } from "@/lib/explainability/ExplainabilityContext";
import "./globals.css";

const manrope = Manrope({
  variable: "--font-heading",
  subsets: ["latin", "cyrillic"],
  weight: ["600", "700", "800"],
});

const inter = Inter({
  variable: "--font-body",
  subsets: ["latin", "cyrillic"],
});

export const metadata: Metadata = {
  title: "HEAN — Beyond Human Emotions",
  description:
    "Autonomous trading intelligence that adapts, evolves, and thrives in any market condition. Anti-fragile architecture with real-time regime detection and genetic strategy evolution.",
  openGraph: {
    title: "HEAN — Beyond Human Emotions",
    description:
      "Autonomous trading intelligence that adapts, evolves, and thrives in any market condition.",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "HEAN — Beyond Human Emotions",
    description:
      "Autonomous trading intelligence that adapts, evolves, and thrives in any market condition.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${manrope.variable} ${inter.variable} antialiased`}
      >
        <ExplainabilityProvider>{children}</ExplainabilityProvider>
      </body>
    </html>
  );
}
