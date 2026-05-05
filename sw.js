const CACHE_NAME = 'nutriscore-v1';
const ASSETS = [
    './',
    './index.html',
    './style.css',
    './app.js',
    'https://unpkg.com/html5-qrcode',
    'https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap'
];

self.addEventListener('install', (e) => {
    e.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
    );
});

self.addEventListener('fetch', (e) => {
    e.respondWith(
        caches.match(e.request).then((response) => response || fetch(e.request))
    );
});

// PWABuilder advanced feature stubs
self.addEventListener('push', (e) => {
    console.log('Push received');
});

self.addEventListener('sync', (e) => {
    console.log('Background sync');
});

self.addEventListener('periodicsync', (e) => {
    console.log('Periodic background sync');
});
