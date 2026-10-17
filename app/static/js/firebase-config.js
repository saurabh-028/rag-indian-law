/* =========================================================
   ShastraShaw — Firebase config
   Public by design (ships in frontend JS anyway) — Firebase's own
   security model relies on server-side token verification, not on
   this config being secret.
   ========================================================= */

const firebaseConfig = {
  apiKey: "AIzaSyDU9eLJqbsBJrnDOAZS02xuvj2LCYNtF-4",
  authDomain: "shastrashaw.firebaseapp.com",
  projectId: "shastrashaw",
  storageBucket: "shastrashaw.firebasestorage.app",
  messagingSenderId: "355447514210",
  appId: "1:355447514210:web:f715a35f2081e8dece9d3b"
};

firebase.initializeApp(firebaseConfig);
const fbAuth = firebase.auth();
