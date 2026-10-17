/* =========================================================
   ShastraShaw — Auth (Firebase sign-in/sign-up/sign-out)
   ========================================================= */

// ── Elements ───────────────────────────────────────────────
const authSignedOut   = document.getElementById('authSignedOut');
const authSignedIn    = document.getElementById('authSignedIn');
const authUserEmail   = document.getElementById('authUserEmail');
const signInBtn       = document.getElementById('signInBtn');
const signOutBtn      = document.getElementById('signOutBtn');

const authModal        = document.getElementById('authModal');
const authModalClose   = document.getElementById('authModalClose');
const authModalError   = document.getElementById('authModalError');
const googleSignInBtn  = document.getElementById('googleSignInBtn');
const authEmailInput   = document.getElementById('authEmailInput');
const authPasswordInput = document.getElementById('authPasswordInput');
const authSignInSubmit = document.getElementById('authSignInSubmit');
const authSignUpSubmit = document.getElementById('authSignUpSubmit');

// ── Modal open/close ───────────────────────────────────────
function openAuthModal() {
  authModalError.style.display = 'none';
  authModalError.textContent = '';
  authModal.classList.add('open');
}

function closeAuthModal() {
  authModal.classList.remove('open');
  authEmailInput.value = '';
  authPasswordInput.value = '';
}

signInBtn?.addEventListener('click', openAuthModal);
authModalClose?.addEventListener('click', closeAuthModal);
authModal?.addEventListener('click', (e) => {
  if (e.target === authModal) closeAuthModal();
});

function showAuthError(err) {
  // Firebase error messages are already reasonably user-facing
  // (e.g. "The email address is already in use by another account.")
  authModalError.textContent = err.message || 'Something went wrong. Please try again.';
  authModalError.style.display = 'block';
}

// ── Sync the signed-in user with our backend ───────────────
// Creates/confirms their row server-side so /query has something to
// attach chat history to. Fire-and-forget from the UI's perspective —
// /query will also lazily create the row if this somehow gets missed.
async function syncUserWithBackend() {
  const user = fbAuth.currentUser;
  if (!user) return;
  try {
    const token = await user.getIdToken();
    await fetch('/auth/sync', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
    });
  } catch (err) {
    console.warn('auth sync failed (non-fatal):', err);
  }
}

// ── Google sign-in ─────────────────────────────────────────
googleSignInBtn?.addEventListener('click', async () => {
  const provider = new firebase.auth.GoogleAuthProvider();
  try {
    await fbAuth.signInWithPopup(provider);
    closeAuthModal();
  } catch (err) {
    showAuthError(err);
  }
});

// ── Email/password sign-in ──────────────────────────────────
authSignInSubmit?.addEventListener('click', async () => {
  const email = authEmailInput.value.trim();
  const password = authPasswordInput.value;
  if (!email || !password) {
    showAuthError({ message: 'Enter both email and password.' });
    return;
  }
  try {
    await fbAuth.signInWithEmailAndPassword(email, password);
    closeAuthModal();
  } catch (err) {
    showAuthError(err);
  }
});

// ── Email/password sign-up ──────────────────────────────────
authSignUpSubmit?.addEventListener('click', async () => {
  const email = authEmailInput.value.trim();
  const password = authPasswordInput.value;
  if (!email || !password) {
    showAuthError({ message: 'Enter both email and password.' });
    return;
  }
  if (password.length < 6) {
    showAuthError({ message: 'Password must be at least 6 characters.' });
    return;
  }
  try {
    await fbAuth.createUserWithEmailAndPassword(email, password);
    closeAuthModal();
  } catch (err) {
    showAuthError(err);
  }
});

// ── Sign out ─────────────────────────────────────────────────
signOutBtn?.addEventListener('click', async () => {
  await fbAuth.signOut();
});

// ── React to sign-in state changing ────────────────────────
fbAuth.onAuthStateChanged((user) => {
  if (user) {
    authSignedOut.style.display = 'none';
    authSignedIn.style.display = 'flex';
    authUserEmail.textContent = user.email || 'Signed in';
    syncUserWithBackend();
  } else {
    authSignedOut.style.display = 'flex';
    authSignedIn.style.display = 'none';
  }
});

// ── Helper used by chat.js: current ID token, or null if signed out ──
async function getAuthToken() {
  const user = fbAuth.currentUser;
  if (!user) return null;
  try {
    return await user.getIdToken();
  } catch (err) {
    console.warn('getIdToken failed:', err);
    return null;
  }
}
