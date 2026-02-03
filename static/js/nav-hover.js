document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('mousemove', e => {
            const rect = item.getBoundingClientRect();
            item.style.setProperty('--x', `${e.clientX - rect.left}px`);
            item.style.setProperty('--y', `${e.clientY - rect.top}px`);
        });
    });
});
