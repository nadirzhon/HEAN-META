/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'hean-dark': '#0a0e27',
        'hean-darker': '#050811',
        'hean-primary': '#00ff88',
        'hean-secondary': '#00d4ff',
        'hean-accent': '#ff006e',
        'hean-warning': '#ffb800',
        'hean-danger': '#ff3366',
        'hean-glow': 'rgba(0, 255, 136, 0.3)',
        'hean-glow-secondary': 'rgba(0, 212, 255, 0.3)',
      },
      fontFamily: {
        mono: ['var(--font-mono)', 'monospace'],
      },
      boxShadow: {
        'glow': '0 0 20px rgba(0, 255, 136, 0.5)',
        'glow-secondary': '0 0 20px rgba(0, 212, 255, 0.5)',
        'glow-danger': '0 0 20px rgba(255, 51, 102, 0.5)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(0, 255, 136, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(0, 255, 136, 0.8)' },
        },
      },
    },
  },
  plugins: [],
};