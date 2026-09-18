// ─────────────────────────────────────────────────────────────────────
//  Tailwind content globs must include the *compiled* SPA output so a
//  production ``npm run build`` (served by FastAPI StaticFiles) keeps
//  every utility that the JSX referenced.
// ─────────────────────────────────────────────────────────────────────
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'Segoe UI', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
