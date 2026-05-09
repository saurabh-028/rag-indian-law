/* =========================================================
   ShastraShaw — Landing Page JS
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

// ── Hero Canvas — Particle System ─────────────────────────
const canvas = document.getElementById('heroCanvas');
const ctx    = canvas ? canvas.getContext('2d') : null;
let particles  = [];
let animFrame;

function resizeCanvas() {
  if (!canvas) return;
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}

class Particle {
  constructor() { this.reset(true); }

  reset(initial = false) {
    this.x         = Math.random() * canvas.width;
    this.y         = initial ? Math.random() * canvas.height : canvas.height + 10;
    this.size      = Math.random() * 1.6 + 0.4;
    this.speedY    = Math.random() * 0.55 + 0.18;
    this.speedX    = (Math.random() - 0.5) * 0.25;
    this.maxOpacity = Math.random() * 0.38 + 0.04;
    this.opacity   = 0;
    this.isGold    = Math.random() < 0.35;
    this.maxLife   = Math.random() * 220 + 120;
    this.life      = initial ? Math.floor(Math.random() * this.maxLife) : 0;
  }

  update() {
    this.y -= this.speedY;
    this.x += this.speedX;
    this.life++;

    const fadeLen = 35;
    if (this.life < fadeLen) {
      this.opacity = (this.life / fadeLen) * this.maxOpacity;
    } else if (this.life > this.maxLife - fadeLen) {
      this.opacity = ((this.maxLife - this.life) / fadeLen) * this.maxOpacity;
    } else {
      this.opacity = this.maxOpacity;
    }

    if (this.life >= this.maxLife || this.y < -10) this.reset();
  }

  draw() {
    ctx.save();
    ctx.globalAlpha = Math.max(0, this.opacity);
    if (this.isGold) {
      ctx.fillStyle  = '#c9a227';
      ctx.shadowBlur = 7;
      ctx.shadowColor = 'rgba(201, 162, 39, 0.5)';
    } else {
      ctx.fillStyle  = '#4f9cf9';
      ctx.shadowBlur = 5;
      ctx.shadowColor = 'rgba(79, 156, 249, 0.35)';
    }
    ctx.beginPath();
    ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

function initParticles() {
  if (!canvas) return;
  particles = [];
  const count = Math.min(Math.floor((canvas.width * canvas.height) / 5500), 90);
  for (let i = 0; i < count; i++) {
    particles.push(new Particle());
  }
}

function animate() {
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  particles.forEach(p => { p.update(); p.draw(); });
  animFrame = requestAnimationFrame(animate);
}

if (canvas) {
  resizeCanvas();
  initParticles();
  animate();

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      resizeCanvas();
      initParticles();
    }, 150);
  }, { passive: true });
}

// ── GSAP Hero entrance animations ─────────────────────────
if (typeof gsap !== 'undefined') {
  const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

  tl.from('.hero-badge',    { duration: 0.8, opacity: 0, y: 24, delay: 0.15 })
    .from('.brand-cinzel',  { duration: 1.0, opacity: 0, y: 36, scale: 0.95 }, '-=0.4')
    .from('.title-sub',     { duration: 0.7, opacity: 0, y: 20 }, '-=0.5')
    .from('.hero-desc',     { duration: 0.7, opacity: 0, y: 18 }, '-=0.45')
    .from('.hero-actions',  { duration: 0.7, opacity: 0, y: 16 }, '-=0.45')
    .from('.hero-stats',    { duration: 0.7, opacity: 0, y: 14 }, '-=0.4')
    .from('.hero-scales-bg',{ duration: 1.2, opacity: 0, x: 30 }, '-=0.9');
}
