/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        forest: {
          DEFAULT: '#0F1F17',
          950: '#0A1610',
        },
        panel: {
          DEFAULT: '#16261D',
          light: '#1C2E23',
          border: '#25392C',
        },
        leaf: {
          DEFAULT: '#6FCF97',
          dim: '#4F9C71',
          glow: '#9BE6BA',
        },
        rust: {
          DEFAULT: '#E8734A',
          dim: '#B85A38',
        },
        bone: '#F1F5F0',
        sage: {
          DEFAULT: '#8FA396',
          dim: '#5E7268',
        },
      },
      fontFamily: {
        heading: ['"Fraunces"', 'ui-serif', 'Georgia', 'serif'],
        body: ['"Inter"', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      boxShadow: {
        panel: '0 0 0 1px #25392C, 0 20px 60px -20px rgba(0,0,0,0.6)',
        glow: '0 0 24px 0 rgba(111, 207, 151, 0.35)',
        rustglow: '0 0 24px 0 rgba(232, 115, 74, 0.35)',
      },
      keyframes: {
        scanline: {
          '0%': { transform: 'translateY(-4%)', opacity: '0.2' },
          '10%': { opacity: '1' },
          '90%': { opacity: '1' },
          '100%': { transform: 'translateY(104%)', opacity: '0.2' },
        },
        reticlePulse: {
          '0%, 100%': { opacity: '0.5', borderColor: 'rgba(111, 207, 151, 0.4)' },
          '50%': { opacity: '1', borderColor: 'rgba(111, 207, 151, 0.95)' },
        },
        flicker: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.85' },
        },
        rise: {
          '0%': { transform: 'scaleX(0)' },
          '100%': { transform: 'scaleX(1)' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        scanline: 'scanline 1.8s cubic-bezier(0.65, 0, 0.35, 1) infinite',
        reticlePulse: 'reticlePulse 1.8s ease-in-out infinite',
        flicker: 'flicker 2.4s ease-in-out infinite',
        'bar-rise': 'rise 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards',
        'fade-up': 'fadeUp 0.4s ease-out forwards',
      },
      letterSpacing: {
        widest2: '0.28em',
      },
      backgroundImage: {
        grid: 'linear-gradient(rgba(143,163,150,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(143,163,150,0.06) 1px, transparent 1px)',
      },
      backgroundSize: {
        grid: '28px 28px',
      },
    },
  },
  plugins: [],
}
