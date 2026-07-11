/* =========================================================
   ShastraShaw: Landing Page JS
   ========================================================= */

// ── AOS init ──────────────────────────────────────────────
AOS.init({
  duration: 720,
  once: true,
  offset: 70,
  easing: 'ease-out-cubic',
});

// ── Navbar scroll effect ───────────────────────────────────
const navbar = document.getElementById('navbar');

window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 60);
}, { passive: true });

// ── Mobile nav toggle ──────────────────────────────────────
const navToggle = document.getElementById('navToggle');
const navLinks  = document.getElementById('navLinks');

navToggle?.addEventListener('click', () => {
  const isOpen = navLinks.classList.toggle('open');
  navToggle.setAttribute('aria-expanded', isOpen);
});

// Close nav when a link is clicked
navLinks?.querySelectorAll('a').forEach(link => {
  link.addEventListener('click', () => navLinks.classList.remove('open'));
});

// ── GSAP Hero entrance animations ─────────────────────────
if (typeof gsap !== 'undefined') {
  const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

  tl.from('.hero-bg',                  { duration: 1.1, opacity: 0 })
    .from('[data-anim="badge"]',       { duration: 0.7, opacity: 0, y: 20 }, '-=0.7')
    .from('[data-anim="title"]',       { duration: 0.9, opacity: 0, y: 30 }, '-=0.4')
    .from('[data-anim="sub"]',         { duration: 0.6, opacity: 0, y: 16 }, '-=0.5')
    .from('[data-anim="desc"]',        { duration: 0.6, opacity: 0, y: 16 }, '-=0.4')
    .from('[data-anim="actions"]',     { duration: 0.6, opacity: 0, y: 14 }, '-=0.4')
    .from('[data-anim="stats"]',       { duration: 0.6, opacity: 0, y: 14 }, '-=0.35');
}
