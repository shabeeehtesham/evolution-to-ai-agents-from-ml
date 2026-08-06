document.addEventListener('DOMContentLoaded', () => {
    /* =====================================================================
       Shared nav active-state highlighting - runs on every page. Each page's
       <body> carries data-page="home|projects|certificates|about", and each
       real nav link carries a matching data-page - no hash-anchor routing
       anymore, these are now separate real pages.
       ===================================================================== */
    const currentPage = document.body.dataset.page;
    if (!currentPage) return;

    document.querySelectorAll('.nav-links a[data-page]').forEach(link => {
        if (link.dataset.page === currentPage) {
            link.classList.add('active');
        }
    });
});
