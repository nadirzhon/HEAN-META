import Hero from "@/sections/Hero";
import Summary from "@/sections/Summary";
import Features from "@/sections/Features";
import Technology from "@/sections/Technology";
import CTA from "@/sections/CTA";
import ClientTour from "@/components/ClientTour";
import ExplainabilityToggle from "@/components/ui/ExplainabilityToggle";

export default function Home() {
  return (
    <main>
      <ExplainabilityToggle />
      <Hero />
      <Summary />
      <section id="features">
        <Features />
      </section>
      <ClientTour />
      <Technology />
      <CTA />
    </main>
  );
}
