/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#0a0a0f',
        'bg-secondary': '#12121a',
        'bg-card': '#1a1a25',
        'text-primary': '#e0e0e8',
        'text-secondary': '#8888a0',
        'accent-green': '#00d4aa',
        'accent-red': '#ff4757',
        'accent-blue': '#4a9eff',
        'accent-yellow': '#ffd43b',
        'border-color': '#2a2a3a',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'IBM Plex Mono', 'Fira Code', 'monospace'],
        sans: ['IBM Plex Sans', 'Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
