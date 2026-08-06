document.addEventListener('DOMContentLoaded', () => {

    /* =====================================================================
       Projects picker - an iOS-picker-style vertical reel. The centered
       item is highlighted; scrolling/dragging through it plays a short
       synthesized tick each time the centered item changes (one tick per
       item crossed, like a UIPickerView); selecting an item shows its real
       preview and links out to its real project page + GitHub source.

       Data-driven: PROJECTS below is the single source of truth for what
       shows up in the reel - add an entry here to add a project, no other
       wiring needed.
       ===================================================================== */
    const PROJECTS = [
        {
            id: 'neurons-to-transformers',
            title: 'Neurons to Transformers',
            tagline: 'The same character-prediction task solved four times, each with a real, better tool: a from-scratch NumPy MLP, a PyTorch RNN/LSTM, a hand-built causal Transformer, and a TF-IDF RAG agent — all four running live in your browser on their actual trained weights.',
            tags: ['NumPy', 'PyTorch', 'Self-Attention', 'RAG'],
            href: 'neurons-to-transformers.html',
            githubHref: 'https://github.com/shabeeehtesham/evolution-to-ai-agents-from-ml',
        },
        // --- TEMPORARY placeholder entries, only here so the scroll/tick
        // interaction has more than one item to try. Not real projects -
        // href is '#' and githubHref points at the profile root, not a
        // fabricated repo. Delete these once real projects are added, or
        // replace with { id, title, tagline, tags, href, githubHref }.
        {
            id: 'placeholder-nebula-forge',
            title: 'Nebula Forge (placeholder)',
            tagline: 'Placeholder entry for testing the picker - replace with a real project.',
            tags: ['placeholder'],
            href: '#',
            githubHref: 'https://github.com/shabeeehtesham',
        },
        {
            id: 'placeholder-quiet-orbit',
            title: 'Quiet Orbit (placeholder)',
            tagline: 'Placeholder entry for testing the picker - replace with a real project.',
            tags: ['placeholder'],
            href: '#',
            githubHref: 'https://github.com/shabeeehtesham',
        },
        {
            id: 'placeholder-copper-atlas',
            title: 'Copper Atlas (placeholder)',
            tagline: 'Placeholder entry for testing the picker - replace with a real project.',
            tags: ['placeholder'],
            href: '#',
            githubHref: 'https://github.com/shabeeehtesham',
        },
        {
            id: 'placeholder-glass-harbor',
            title: 'Glass Harbor (placeholder)',
            tagline: 'Placeholder entry for testing the picker - replace with a real project.',
            tags: ['placeholder'],
            href: '#',
            githubHref: 'https://github.com/shabeeehtesham',
        },
        // Future real projects: append { id, title, tagline, tags, href, githubHref } here.
    ];

    const reel = document.getElementById('picker-reel');
    const previewEl = document.getElementById('picker-preview');
    if (!reel || !previewEl) return;

    const items = [];
    let centeredIndex = -1;
    let audioCtx = null;

    function ensureAudio() {
        if (!audioCtx) {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (!Ctx) return;
            audioCtx = new Ctx();
        }
        if (audioCtx.state === 'suspended') audioCtx.resume();
    }

    // A short synthesized click - no audio file needed. Fired only when the
    // centered item changes, so exactly one tick plays per item crossed.
    // This is a filtered noise burst (18ms of white noise through a
    // bandpass filter around ~2.2-3kHz with a fast exponential gain
    // envelope), not an oscillator tone - a pure tone reads as a "beep,"
    // filtered noise reads as a mechanical "tick," which is what an iOS
    // picker actually sounds like.
    function playTick() {
        if (!audioCtx) return;
        const now = audioCtx.currentTime;
        const duration = 0.018;
        const bufferLen = Math.max(1, Math.floor(audioCtx.sampleRate * duration));
        const buffer = audioCtx.createBuffer(1, bufferLen, audioCtx.sampleRate);
        const data = buffer.getChannelData(0);
        for (let i = 0; i < bufferLen; i++) {
            data[i] = (Math.random() * 2 - 1) * (1 - i / bufferLen); // decaying white noise
        }

        const source = audioCtx.createBufferSource();
        source.buffer = buffer;

        const filter = audioCtx.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.setValueAtTime(2200 + Math.random() * 800, now);
        filter.Q.value = 1.2;

        const gain = audioCtx.createGain();
        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.exponentialRampToValueAtTime(0.55, now + 0.001);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);

        source.connect(filter);
        filter.connect(gain);
        gain.connect(audioCtx.destination);
        source.start(now);
        source.stop(now + duration + 0.01);
    }

    function renderPreview(project) {
        previewEl.innerHTML = `
            <h3>${project.title}</h3>
            <p>${project.tagline}</p>
            <div class="picker-tags">${project.tags.map(t => `<span class="project-tag">${t}</span>`).join('')}</div>
            <div class="picker-preview-actions">
                <a href="${project.href}" class="btn btn-primary btn-sm">View Project</a>
                <a href="${project.githubHref}" target="_blank" class="btn btn-secondary btn-sm">GitHub &#8599;</a>
            </div>
        `;
    }

    function scrollToIndex(idx, smooth = true) {
        const item = items[idx];
        if (!item) return;
        item.scrollIntoView({ block: 'center', behavior: smooth ? 'smooth' : 'auto', inline: 'nearest' });
    }

    // Finds the item nearest the reel's vertical center, applies a fisheye
    // dim/scale to the rest, and - only when the centered item actually
    // changes - plays a tick and swaps the preview panel.
    function updateCenteredItem() {
        const reelRect = reel.getBoundingClientRect();
        const centerY = reelRect.top + reelRect.height / 2;
        let closestIdx = 0;
        let closestDist = Infinity;

        items.forEach((item, idx) => {
            // transform:scale() is used (not font-size) specifically because
            // it doesn't affect layout - scaling from the element's own
            // center leaves its measured center point stable, so this
            // distance calculation doesn't feed back on itself. An earlier
            // version changed font-size directly, which changes each item's
            // rendered height and therefore its position in the scroll
            // flow - shifting positions while they were being measured,
            // which is what caused the scroll jitter.
            const r = item.getBoundingClientRect();
            const itemCenter = r.top + r.height / 2;
            const dist = Math.abs(itemCenter - centerY);
            if (dist < closestDist) { closestDist = dist; closestIdx = idx; }

            const norm = Math.min(1, dist / (reelRect.height / 2 || 1));
            item.style.opacity = String(1 - norm * 0.75);
            item.style.transform = `scale(${(1.45 - norm * 0.85).toFixed(3)})`;
        });

        if (closestIdx !== centeredIndex) {
            if (centeredIndex !== -1) playTick(); // no tick on the initial, pre-interaction placement
            centeredIndex = closestIdx;
            items.forEach((item, idx) => {
                item.classList.toggle('active', idx === closestIdx);
                item.setAttribute('aria-selected', idx === closestIdx ? 'true' : 'false');
            });
            renderPreview(PROJECTS[closestIdx]);
        }
    }

    function buildReel() {
        const topSpacer = document.createElement('div');
        topSpacer.className = 'picker-spacer';
        reel.appendChild(topSpacer);

        PROJECTS.forEach((project, idx) => {
            const item = document.createElement('div');
            item.className = 'picker-item';
            item.textContent = project.title;
            item.dataset.idx = String(idx);
            item.setAttribute('role', 'option');
            item.setAttribute('aria-selected', 'false');
            item.addEventListener('click', () => {
                ensureAudio();
                scrollToIndex(idx, true);
            });
            reel.appendChild(item);
            items.push(item);
        });

        const bottomSpacer = document.createElement('div');
        bottomSpacer.className = 'picker-spacer';
        reel.appendChild(bottomSpacer);

        // Spacers sized to half the reel's visible height so the first/last
        // real items can still scroll all the way to center.
        requestAnimationFrame(() => {
            const reelHeight = reel.clientHeight;
            const itemHeight = items[0] ? items[0].offsetHeight : 64;
            const spacerHeight = Math.max(0, (reelHeight - itemHeight) / 2);
            topSpacer.style.height = `${spacerHeight}px`;
            bottomSpacer.style.height = `${spacerHeight}px`;

            scrollToIndex(0, false);
            updateCenteredItem();
        });
    }

    let scrollRaf = null;
    reel.addEventListener('scroll', () => {
        if (scrollRaf) return;
        scrollRaf = requestAnimationFrame(() => {
            updateCenteredItem();
            scrollRaf = null;
        });
    }, { passive: true });

    // AudioContext creation is gated to real user-gesture events (never the
    // scroll event itself, which also fires for programmatic/smooth scrolls)
    // to respect browser autoplay policy.
    reel.addEventListener('wheel', ensureAudio, { passive: true });
    reel.addEventListener('touchstart', ensureAudio, { passive: true });
    reel.addEventListener('mousedown', ensureAudio);

    reel.setAttribute('tabindex', '0');
    reel.setAttribute('role', 'listbox');
    reel.addEventListener('keydown', (e) => {
        ensureAudio();
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            scrollToIndex(Math.min(items.length - 1, centeredIndex + 1));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            scrollToIndex(Math.max(0, centeredIndex - 1));
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (centeredIndex >= 0) window.location.href = PROJECTS[centeredIndex].href;
        }
    });

    buildReel();
});
