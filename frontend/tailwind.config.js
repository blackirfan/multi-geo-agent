/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{html,ts}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        navy:   { DEFAULT: '#0f3460', light: '#16213e', dark: '#0a1f3d' },
        crimson:{ DEFAULT: '#e94560', light: '#f06b82', dark: '#c73350' },
        slate:  { 850: '#1a2035' },
      },
      animation: {
        'pulse-ring': 'pulse-ring 1.4s cubic-bezier(0.4,0,0.6,1) infinite',
      },
      keyframes: {
        'pulse-ring': {
          '0%, 100%': { opacity: '0.8', transform: 'scale(1)' },
          '50%':       { opacity: '0.3', transform: 'scale(1.18)' },
        },
      },
    },
  },
  plugins: [],
};
