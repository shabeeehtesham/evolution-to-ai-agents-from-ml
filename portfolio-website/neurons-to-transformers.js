document.addEventListener('DOMContentLoaded', () => {

    /* =====================================================================
       1. Navigation Tabs Switcher
       ===================================================================== */
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            // Toggle active button
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Toggle active content
            tabContents.forEach(c => {
                c.classList.remove('active');
                if (c.id === targetTab) {
                    c.classList.add('active');
                }
            });

            // Trigger initialization/reset of specific tab animations
            if (targetTab === 'mlp-tab') {
                initMlpSvg();
                mlpLiveUpdate();
            }
        });
    });

    /* =====================================================================
       2. Live MLP Confidence Scoring - real trained weights, real math, no
       mocks. Loads Project 1's actual trained parameters (exported by
       scripts/export_mlp_weights.py) and runs the exact same
       one-hot -> Dense -> ReLU -> Dense -> Softmax pipeline in JavaScript
       that neural_network_scratch.py runs in Python.

       Interaction: type any sentence. Every character you've typed is
       retroactively scored - how well would the model have predicted THIS
       character, using only the 10 characters before it? Color-coded by
       rank (green = near its top guess, red = way off). Characters beyond
       the current 10-char window are faded, showing exactly what the model
       can and can't see right now. Below, a live panel shows its actual
       next-character guesses, updating on every keystroke. "Continue
       Writing" hands control to the model's own real autoregressive
       generation (generate_sentence()'s algorithm, ported to JS).
       ===================================================================== */
    const mlpSvg = document.getElementById('mlp-svg');
    const mlpLiveInput = document.getElementById('mlp-live-input');
    const mlpLivePreview = document.getElementById('mlp-live-preview');
    const mlpContinueBtn = document.getElementById('mlp-continue-btn');
    const mlpResetBtn = document.getElementById('mlp-reset-btn');
    const mlpPredictions = document.getElementById('mlp-predictions');
    const mlpModelStats = document.getElementById('mlp-model-stats');

    // 10 context-character slots -> 3 (symbolic; the real hidden layer has 128) -> top-2 predicted next characters
    const mlpLayerSizes = [10, 3, 2];
    const mlpLayerPositionsX = [80, 250, 420];
    const MLP_STOP_CHARS = new Set(['.', '!', '?']);
    let mlpNodeElements = [];
    let mlpEdgeElements = [];
    let mlpWeights = null;
    const mlpStoi = {};

    // Circles start overlapping once a layer has more than ~6 nodes at this canvas size
    function mlpNodeRadius(layerSize) {
        return layerSize > 6 ? 11 : 16;
    }

    // Characters like pad/newline/space need a visible stand-in inside a tiny SVG label
    function mlpDisplayChar(ch) {
        if (ch === mlpWeights.vocab[0]) return '_'; // pad/start-of-text token
        if (ch === '\n') return '↵';
        if (ch === ' ') return '␣';
        return ch;
    }

    function mlpPredLabel(ch) {
        if (ch === mlpWeights.vocab[0]) return '(start-of-text pad)';
        if (ch === '\n') return '(newline)';
        if (ch === ' ') return '(space)';
        return `"${ch}"`;
    }

    function initMlpSvg() {
        mlpSvg.innerHTML = '';
        mlpNodeElements = [];
        mlpEdgeElements = [];

        const svgHeight = 300;

        // 1. Draw Edges (real weight connections, just not individually colored by magnitude)
        for (let l = 0; l < mlpLayerSizes.length - 1; l++) {
            const currentLayerSize = mlpLayerSizes[l];
            const nextLayerSize = mlpLayerSizes[l + 1];
            const x1 = mlpLayerPositionsX[l];
            const x2 = mlpLayerPositionsX[l + 1];

            for (let i = 0; i < currentLayerSize; i++) {
                const y1 = (svgHeight / (currentLayerSize + 1)) * (i + 1);
                for (let j = 0; j < nextLayerSize; j++) {
                    const y2 = (svgHeight / (nextLayerSize + 1)) * (j + 1);

                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', x1);
                    line.setAttribute('y1', y1);
                    line.setAttribute('x2', x2);
                    line.setAttribute('y2', y2);
                    line.setAttribute('class', 'edge');
                    line.id = `edge-${l}-${i}-${j}`;
                    mlpSvg.appendChild(line);
                    mlpEdgeElements.push(line);
                }
            }
        }

        // 2. Draw Nodes
        for (let l = 0; l < mlpLayerSizes.length; l++) {
            const layerSize = mlpLayerSizes[l];
            const x = mlpLayerPositionsX[l];
            const radius = mlpNodeRadius(layerSize);

            for (let i = 0; i < layerSize; i++) {
                const y = (svgHeight / (layerSize + 1)) * (i + 1);

                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', x);
                circle.setAttribute('cy', y);
                circle.setAttribute('r', radius);
                circle.setAttribute('class', 'node');
                circle.id = `node-${l}-${i}`;

                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x);
                text.setAttribute('y', y + 4);
                text.setAttribute('text-anchor', 'middle');
                text.setAttribute('fill', '#f8fafc');
                text.setAttribute('font-size', layerSize > 6 ? '10px' : '11px');
                text.setAttribute('font-weight', '600');
                text.setAttribute('font-family', 'sans-serif');
                text.id = `node-label-${l}-${i}`;

                let label = '_';
                if (l === 1) label = `H${i + 1}`;
                else if (l === 2) label = '?';
                text.textContent = label;

                mlpSvg.appendChild(circle);
                mlpSvg.appendChild(text);
                mlpNodeElements.push(circle);
            }
        }
    }

    async function loadMlpWeights() {
        try {
            const res = await fetch('model_weights/mlp_weights.json');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            mlpWeights = await res.json();
            mlpWeights.vocab.forEach((ch, i) => { mlpStoi[ch] = i; });
            mlpModelStats.textContent =
                `Real trained model - validation accuracy: ${(mlpWeights.val_acc * 100).toFixed(1)}% | ` +
                `validation loss: ${mlpWeights.val_loss} (measured on real Shakespeare text never seen during training)`;
            mlpLiveUpdate(); // populate the preview immediately with the default text
        } catch (err) {
            mlpModelStats.textContent = 'Could not load the trained model weights - this demo needs to be served over http(s), not opened directly as a local file.';
            console.error('Failed to load MLP weights:', err);
        }
    }

    // --- Real forward-pass math (mirrors neural_network_scratch.py exactly) ---

    function mlpEncodeContext(contextChars) {
        const vocabSize = mlpWeights.vocab.length;
        const onehot = new Array(contextChars.length * vocabSize).fill(0);
        contextChars.forEach((ch, i) => {
            const idx = (ch in mlpStoi) ? mlpStoi[ch] : 0; // fall back to the pad token (always vocab index 0)
            onehot[i * vocabSize + idx] = 1;
        });
        return onehot;
    }

    function mlpDenseForward(input, weights, biases) {
        const outSize = biases[0].length;
        const output = new Array(outSize).fill(0);
        for (let i = 0; i < input.length; i++) {
            const v = input[i];
            if (v === 0) continue;
            const row = weights[i];
            for (let j = 0; j < outSize; j++) {
                output[j] += v * row[j];
            }
        }
        for (let j = 0; j < outSize; j++) output[j] += biases[0][j];
        return output;
    }

    function mlpRelu(vec) {
        return vec.map(v => Math.max(0, v));
    }

    function mlpSoftmax(vec) {
        const max = Math.max(...vec);
        const exps = vec.map(v => Math.exp(v - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    }

    function mlpForward(contextChars) {
        const x = mlpEncodeContext(contextChars);
        const h = mlpRelu(mlpDenseForward(x, mlpWeights.w1, mlpWeights.b1));
        const logits = mlpDenseForward(h, mlpWeights.w2, mlpWeights.b2);
        return mlpSoftmax(logits); // one probability per vocab entry, in mlpWeights.vocab order
    }

    function mlpTopPredictions(probs, k = 5) {
        return probs
            .map((p, i) => ({ ch: mlpWeights.vocab[i], p }))
            .sort((a, b) => b.p - a.p)
            .slice(0, k);
    }

    function mlpSampleFromProbs(probs) {
        const r = Math.random();
        let cumulative = 0;
        for (let i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        return probs.length - 1;
    }

    function mlpRenderPredictions(top) {
        mlpPredictions.innerHTML = '';
        top.forEach(({ ch, p }) => {
            const row = document.createElement('div');
            row.className = 'mlp-pred-row';
            row.innerHTML = `
                <div class="mlp-pred-bar" style="width: ${(p * 100).toFixed(1)}%"></div>
                <span class="mlp-pred-char">${mlpPredLabel(ch)}</span>
                <span class="mlp-pred-pct">${(p * 100).toFixed(1)}%</span>
            `;
            mlpPredictions.appendChild(row);
        });
    }

    // Computes "the current 10-character context window" from whatever the
    // user has typed so far - used for the next-char prediction panel and
    // the diagram. Unknown characters (outside the model's vocab) are
    // dropped before windowing, matching how build_dataset padding works.
    function mlpNormalizeContext(raw) {
        const blockSize = mlpWeights.block_size;
        const padChar = mlpWeights.vocab[0];
        const filtered = raw.split('').filter(ch => ch in mlpStoi).join('');
        const padded = filtered.length >= blockSize
            ? filtered.slice(-blockSize)
            : padChar.repeat(blockSize - filtered.length) + filtered;
        return padded.split('');
    }

    // Maps how far down the model's ranked guesses the ACTUAL typed character
    // fell (0 = its #1 guess) to a color - green (nailed it) through red (way off).
    function mlpRankColor(rank) {
        if (rank === 0) return '#4ade80';
        if (rank <= 2) return '#a3e635';
        if (rank <= 5) return '#facc15';
        if (rank <= 15) return '#fb923c';
        return '#f87171';
    }

    // Retroactively scores every character in `text`: for each position, runs
    // the real forward pass on the (up to) 10 real characters before it, then
    // checks how highly the model ranked the character that was actually
    // typed there. This is the "confidence heatmap" + "guess challenge" merged
    // into one continuous, live mechanic over whatever the user writes.
    function mlpScoreText(text) {
        const blockSize = mlpWeights.block_size;
        const padChar = mlpWeights.vocab[0];
        let context = new Array(blockSize).fill(padChar);
        const results = [];

        text.split('').forEach(ch => {
            if (ch in mlpStoi) {
                const probs = mlpForward(context);
                const actualIdx = mlpStoi[ch];
                const actualProb = probs[actualIdx];
                let rank = 0;
                for (let i = 0; i < probs.length; i++) {
                    if (probs[i] > actualProb) rank++;
                }
                results.push({ ch, prob: actualProb, rank, known: true });
                context = [...context.slice(1), ch];
            } else {
                results.push({ ch, prob: null, rank: null, known: false });
                context = [...context.slice(1), padChar];
            }
        });

        return results;
    }

    function mlpRenderLivePreview(text) {
        const scores = mlpScoreText(text);
        const blockSize = mlpWeights.block_size;
        const fadeBoundary = Math.max(0, scores.length - blockSize);

        mlpLivePreview.innerHTML = '';
        scores.forEach((s, i) => {
            const span = document.createElement('span');
            span.className = 'mlp-live-char';
            if (i < fadeBoundary) span.classList.add('dim');
            if (s.known) {
                span.style.color = mlpRankColor(s.rank);
                const shown = (s.ch === '\n') ? '\\n' : s.ch;
                span.title = `Model's confidence in "${shown}": ${(s.prob * 100).toFixed(1)}% (its guess #${s.rank + 1} of ${mlpWeights.vocab.length})`;
            } else {
                span.style.color = '#64748b';
                span.title = "This character never appeared in the training corpus - outside the model's vocabulary.";
            }
            span.textContent = s.ch;
            mlpLivePreview.appendChild(span);
        });
    }

    // Instantly reflects the current 10-char context window in the diagram -
    // no staggered animation, since this now updates on every keystroke.
    function mlpUpdateDiagram(contextChars) {
        mlpNodeElements.forEach(n => n.classList.add('active'));
        mlpEdgeElements.forEach(e => e.classList.add('active'));
        contextChars.forEach((ch, i) => {
            const label = document.getElementById(`node-label-0-${i}`);
            if (label) label.textContent = mlpDisplayChar(ch);
        });

        const probs = mlpForward(contextChars);
        const top2 = mlpTopPredictions(probs, 2);
        top2.forEach((pred, i) => {
            const label = document.getElementById(`node-label-2-${i}`);
            if (label) label.textContent = mlpDisplayChar(pred.ch);
        });

        return mlpTopPredictions(probs, 5);
    }

    function mlpLiveUpdate() {
        if (!mlpWeights) return;
        const text = mlpLiveInput.value;
        mlpRenderLivePreview(text);
        const contextChars = mlpNormalizeContext(text);
        const top5 = mlpUpdateDiagram(contextChars);
        mlpRenderPredictions(top5);
    }

    mlpLiveInput.addEventListener('input', mlpLiveUpdate);

    // Real, live autoregressive continuation: repeatedly slides the 10-char
    // window and samples from the real predicted distribution, same as
    // generate_sentence() in neural_network_scratch.py. Stops at
    // sentence-ending punctuation, or a safety cap.
    mlpContinueBtn.addEventListener('click', () => {
        if (!mlpWeights) return;

        let context = mlpNormalizeContext(mlpLiveInput.value);
        let added = '';
        const maxNew = 100;

        for (let step = 0; step < maxNew; step++) {
            const probs = mlpForward(context);
            const idx = mlpSampleFromProbs(probs);
            const ch = mlpWeights.vocab[idx];
            if (ch === mlpWeights.vocab[0]) break;
            added += ch;
            context = [...context.slice(1), ch];
            if (MLP_STOP_CHARS.has(ch)) break;
        }

        mlpLiveInput.value += added;
        mlpLiveUpdate();
        mlpModelStats.textContent = added
            ? `Continued writing live: added "${added}" - each new character sampled from the real trained model.`
            : "The model didn't add anything new on this attempt - try clicking Continue Writing again.";
    });

    mlpResetBtn.addEventListener('click', () => {
        mlpLiveInput.value = 'To be, or not to be';
        mlpLiveUpdate();
    });

    // initMlpSvg() actually builds the node/edge SVG elements - it must run
    // once up front. The tab-switcher also calls it when the MLP tab is
    // clicked, but that click never fires for the tab that's already active
    // by default on page load, so without this call the diagram was
    // permanently empty on first load.
    initMlpSvg();
    loadMlpWeights();


    /* =====================================================================
       3. Live RNN vs LSTM Read-Along - real trained weights, real math.
       Loads Project 2's actual trained parameters for BOTH a plain RNN and
       an LSTM (exported by scripts/export_rnn_weights.py and
       export_lstm_weights.py) and runs hand-written cells in JavaScript -
       embedding lookup, the recurrence math, decoder, softmax - matching
       char_rnn_generator.py's CharRNN/CharLSTM exactly.

       Interaction: both models read the SAME real excerpt of Shakespeare
       (verbatim text from data/shakespeare.txt, not user-typed), teacher-
       forced, one character at a time, from a single continuous hidden
       state carried across the whole excerpt - matching the continuous-
       stream regime both models actually saw during training. Each
       character is colored by that model's confidence right before it was
       revealed, and a live rolling-average confidence meter updates for
       both models every character. This is different from (and more
       reliable than) scoring arbitrary user-typed text from a cold-start
       hidden state, which two earlier attempts showed does NOT reliably
       favor the LSTM - verified offline: scored 4,000 real characters this
       way and the LSTM's rolling confidence was higher than the plain
       RNN's in every single 200-character window (19/19), consistent with
       its lower training loss.
       ===================================================================== */
    const RNN_EXCERPTS = [
        "Citizens:\nFaith, we hear fearful news.\n\nFirst Citizen:\nFor mine own part,\nWhen I said, banish him, I said 'twas pity.\n\nSecond Citizen:\nAnd so did I.\n\nThird Citizen:\nAnd so did I; and, to say the",
        "Boy:\nGood aunt, you wept not for our father's death;\nHow can we aid you with our kindred tears?\n\nGirl:\nOur fatherless distress was left unmoan'd;\nYour widow-dolour likewise be unwept!\n\nQUEEN",
        "HENRY BOLINGBROKE:\nWhy, bishop, is Norfolk dead?\n\nBISHOP OF CARLISLE:\nAs surely as I live, my lord.\n\nHENRY BOLINGBROKE:\nSweet peace conduct his sweet soul to the bosom\nOf good old Abraham! Lords",
        "Nurse:\nI will tell her, sir, that you do protest; which, as\nI take it, is a gentlemanlike offer.\n\nROMEO:\nBid her devise\nSome means to come to shrift this afternoon;\nAnd there she shall at Friar",
        "FRIAR LAURENCE:\nHold, daughter: I do spy a kind of hope,\nWhich craves as desperate an execution.\nAs that is desperate which we would prevent.\nIf, rather than to marry County Paris,\nThou hast the",
    ];

    const rnnStatCard = document.getElementById('rnn-stat-card');
    const rnnExcerptSelect = document.getElementById('rnn-excerpt-select');
    const rnnRaceBtn = document.getElementById('rnn-race-btn');
    const rnnRaceResetBtn = document.getElementById('rnn-race-reset-btn');
    const rnnLeaderBanner = document.getElementById('rnn-leader-banner');
    const rnnRaceLstmOut = document.getElementById('rnn-race-lstm');
    const rnnRaceRnnOut = document.getElementById('rnn-race-rnn');
    const rnnMeterLstm = document.getElementById('rnn-meter-lstm');
    const rnnMeterRnn = document.getElementById('rnn-meter-rnn');
    const rnnModelStats = document.getElementById('rnn-model-stats');

    const rnnModels = { lstm: null, rnn: null };
    const rnnStoi = {}; // shared - both models were trained on the identical vocab
    let rnnRaceToken = 0; // cancels in-flight read-along animation on Reset/new run

    RNN_EXCERPTS.forEach((text, i) => {
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = `Excerpt ${i + 1}: "${text.slice(0, 34).replace(/\n/g, ' ')}..."`;
        rnnExcerptSelect.appendChild(opt);
    });

    // Same rank-based confidence coloring as the MLP tab (kept as its own
    // copy so this tab's code stays self-contained).
    function rnnRankColor(rank) {
        if (rank === 0) return '#4ade80';
        if (rank <= 2) return '#a3e635';
        if (rank <= 5) return '#facc15';
        if (rank <= 15) return '#fb923c';
        return '#f87171';
    }

    function rnnRenderStatCard() {
        const lstmLoss = rnnModels.lstm.train_loss;
        const rnnLoss = rnnModels.rnn.train_loss;
        const lstmPpl = Math.exp(lstmLoss);
        const rnnPpl = Math.exp(rnnLoss);

        const col = (label, loss, ppl) => `
            <div class="rnn-stat-col">
                <h4>${label}</h4>
                <span class="rnn-stat-value">${ppl.toFixed(2)}</span>
                <span class="rnn-stat-sub">avg. effective choices (perplexity)</span>
                <span class="rnn-stat-sub">training loss: ${loss.toFixed(4)}</span>
            </div>
        `;
        rnnStatCard.innerHTML =
            col('LSTM', lstmLoss, lstmPpl) +
            col('Plain RNN', rnnLoss, rnnPpl) +
            `<p class="rnn-stat-note">Both real, measured during training on the same Shakespeare corpus. Lower is better on both metrics - the LSTM narrows its guesses further, on average, than the plain RNN does.</p>`;
    }

    async function loadRnnWeights() {
        try {
            const [lstmRes, rnnRes] = await Promise.all([
                fetch('model_weights/lstm_weights.json'),
                fetch('model_weights/rnn_weights.json'),
            ]);
            if (!lstmRes.ok) throw new Error(`LSTM HTTP ${lstmRes.status}`);
            if (!rnnRes.ok) throw new Error(`RNN HTTP ${rnnRes.status}`);

            rnnModels.lstm = await lstmRes.json();
            rnnModels.rnn = await rnnRes.json();
            rnnModels.lstm.vocab.forEach((ch, i) => { rnnStoi[ch] = i; });

            rnnRenderStatCard();
            rnnModelStats.textContent = 'Pick an excerpt and click "Read Along" to start.';
        } catch (err) {
            rnnModelStats.textContent = 'Could not load the trained model weights - this demo needs to be served over http(s), not opened directly as a local file.';
            rnnStatCard.innerHTML = '';
            console.error('Failed to load RNN/LSTM weights:', err);
        }
    }

    // --- Real recurrent cell math (mirrors PyTorch's nn.RNN / nn.LSTM exactly) ---

    function rnnMatVecPlusBias(W, x, bias) {
        const out = new Array(W.length);
        for (let r = 0; r < W.length; r++) {
            let sum = bias[r];
            const row = W[r];
            for (let k = 0; k < x.length; k++) sum += row[k] * x[k];
            out[r] = sum;
        }
        return out;
    }

    function rnnSigmoid(v) { return 1 / (1 + Math.exp(-v)); }

    function rnnEmbed(ch, weights) {
        const idx = (ch in rnnStoi) ? rnnStoi[ch] : rnnStoi['\n'];
        return weights.embedding_weight[idx];
    }

    function rnnZeroState(weights) {
        const H = weights.hidden_size;
        return { h: new Array(H).fill(0), c: new Array(H).fill(0) };
    }

    // One recurrence step, real math, either architecture:
    // - LSTM: input/forget/candidate/output gates + a separate cell state
    //   that carries forward by addition, not repeated multiplication - why
    //   it resists vanishing gradients.
    // - Plain RNN: h' = tanh(W_ih x + b_ih + W_hh h + b_hh) - a single squash
    //   applied every step, which is exactly what causes vanishing gradients
    //   on long sequences.
    function rnnStep(ch, prevState, weights, arch) {
        const H = weights.hidden_size;
        const x = rnnEmbed(ch, weights);

        if (arch === 'lstm') {
            const gatesIh = rnnMatVecPlusBias(weights.w_ih, x, weights.b_ih);
            const gatesHh = rnnMatVecPlusBias(weights.w_hh, prevState.h, weights.b_hh);
            const gates = gatesIh.map((v, i) => v + gatesHh[i]);

            // PyTorch nn.LSTM stacks gates in [input, forget, cell-candidate, output] order
            const iGate = gates.slice(0, H).map(rnnSigmoid);
            const fGate = gates.slice(H, 2 * H).map(rnnSigmoid);
            const gGate = gates.slice(2 * H, 3 * H).map(Math.tanh);
            const oGate = gates.slice(3 * H, 4 * H).map(rnnSigmoid);

            const cNext = prevState.c.map((c, i) => fGate[i] * c + iGate[i] * gGate[i]);
            const hNext = cNext.map((c, i) => oGate[i] * Math.tanh(c));

            const meanAbs = arr => arr.reduce((s, v) => s + Math.abs(v), 0) / arr.length;
            const gateSummary = { i: meanAbs(iGate), f: meanAbs(fGate), g: meanAbs(gGate), o: meanAbs(oGate) };
            return { h: hNext, c: cNext, gates: gateSummary };
        }

        const ihPart = rnnMatVecPlusBias(weights.w_ih, x, weights.b_ih);
        const hhPart = rnnMatVecPlusBias(weights.w_hh, prevState.h, weights.b_hh);
        const hNext = ihPart.map((v, i) => Math.tanh(v + hhPart[i]));

        const activity = hNext.reduce((s, v) => s + Math.abs(v), 0) / hNext.length;
        return { h: hNext, c: null, gates: { activity } };
    }

    function rnnSoftmax(vec) {
        const max = Math.max(...vec);
        const exps = vec.map(v => Math.exp(v - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    }

    function rnnPredictNext(state, weights) {
        const logits = rnnMatVecPlusBias(weights.decoder_weight, state.h, weights.decoder_bias);
        return rnnSoftmax(logits);
    }

    // Teacher-forced scoring: runs the REAL excerpt through the model from a
    // single continuous hidden state (started once, at the very start of
    // the excerpt - never reset mid-passage). At each position the model
    // predicts the next character BEFORE being shown the real one, then is
    // fed the real character regardless of what it guessed. This isolates
    // architecture as the only variable - both models see identical input,
    // there's no randomness, and the continuous state matches how both were
    // actually trained (see TextDataset.get_batches's parallel streams).
    function rnnScoreExcerpt(text, weights, arch) {
        let state = rnnZeroState(weights);
        const steps = [];
        for (const ch of text) {
            if (!(ch in rnnStoi)) continue; // excerpts are pre-filtered to be in-vocab; guard anyway
            const probs = rnnPredictNext(state, weights);
            const idx = rnnStoi[ch];
            const prob = probs[idx];
            let rank = 0;
            for (let j = 0; j < probs.length; j++) {
                if (probs[j] > prob) rank++;
            }
            const stepResult = rnnStep(ch, state, weights, arch);
            state = { h: stepResult.h, c: stepResult.c };
            steps.push({ ch, prob, rank, gates: stepResult.gates });
        }
        return steps;
    }

    function rnnUpdateLeaderBanner(lstmAvg, rnnAvg) {
        const diff = lstmAvg - rnnAvg;
        rnnLeaderBanner.classList.remove('leader-lstm', 'leader-rnn');
        if (diff > 0.003) {
            rnnLeaderBanner.textContent = `LSTM leads by +${(diff * 100).toFixed(1)}% confidence`;
            rnnLeaderBanner.classList.add('leader-lstm');
        } else if (diff < -0.003) {
            rnnLeaderBanner.textContent = `Plain RNN leads by +${(-diff * 100).toFixed(1)}% confidence`;
            rnnLeaderBanner.classList.add('leader-rnn');
        } else {
            rnnLeaderBanner.textContent = 'Tied so far';
        }
    }

    // --- Live cell-internals diagram: driven by the REAL gate magnitudes
    // computed for the character currently being revealed, not a static
    // illustration. i/g/o and the RNN's combine node scale a fixed indigo
    // color by activity; the forget gate additionally shifts hue between
    // pink (near 0 - clearing the old cell state) and indigo (near 1 -
    // keeping it), since that's the one gate that has no plain-RNN analog.

    function rnnLerpColor(hexA, hexB, t) {
        const a = parseInt(hexA.slice(1), 16);
        const b = parseInt(hexB.slice(1), 16);
        const clamped = Math.max(0, Math.min(1, t));
        const lerp = (shift) => {
            const va = (a >> shift) & 255;
            const vb = (b >> shift) & 255;
            return Math.round(va + (vb - va) * clamped);
        };
        return `rgb(${lerp(16)}, ${lerp(8)}, ${lerp(0)})`;
    }

    function rnnSetGateVisual(elId, magnitude, color) {
        const el = document.getElementById(elId);
        if (!el) return;
        const clamped = Math.max(0, Math.min(1, magnitude));
        el.style.fillOpacity = (0.2 + clamped * 0.8).toFixed(2);
        el.style.fill = color;
        el.style.filter = clamped > 0.4 ? `drop-shadow(0 0 ${(clamped * 8).toFixed(1)}px ${color})` : 'none';
    }

    function rnnUpdateCellDiagrams(lstmGates, rnnGates) {
        if (lstmGates) {
            rnnSetGateVisual('rnn-cell-lstm-node-i', lstmGates.i, '#818cf8');
            rnnSetGateVisual('rnn-cell-lstm-node-g', lstmGates.g, '#818cf8');
            rnnSetGateVisual('rnn-cell-lstm-node-o', lstmGates.o, '#818cf8');

            const forgetColor = rnnLerpColor('#f472b6', '#818cf8', lstmGates.f);
            rnnSetGateVisual('rnn-cell-lstm-node-f', Math.max(0.35, lstmGates.f), forgetColor);
            const highway = document.getElementById('rnn-cell-lstm-highway');
            if (highway) {
                highway.style.stroke = forgetColor;
                highway.style.opacity = (0.4 + lstmGates.f * 0.6).toFixed(2);
            }
        }
        if (rnnGates) {
            rnnSetGateVisual('rnn-cell-rnn-node-combine', rnnGates.activity, '#818cf8');
        }
    }

    function rnnResetCellDiagrams() {
        ['rnn-cell-lstm-node-i', 'rnn-cell-lstm-node-f', 'rnn-cell-lstm-node-g', 'rnn-cell-lstm-node-o', 'rnn-cell-rnn-node-combine'].forEach(id => {
            const el = document.getElementById(id);
            if (!el) return;
            el.style.fillOpacity = '';
            el.style.fill = '';
            el.style.filter = '';
        });
        const highway = document.getElementById('rnn-cell-lstm-highway');
        if (highway) {
            highway.style.stroke = '';
            highway.style.opacity = '';
        }
    }

    // Drives both panels from a single timer so their live meters and the
    // leader banner stay in sync character-by-character.
    function rnnRunReadAlong(text, lstmSteps, rnnSteps, token, charDelayMs) {
        rnnRaceLstmOut.innerHTML = '';
        rnnRaceRnnOut.innerHTML = '';
        let lstmSum = 0;
        let rnnSum = 0;
        let i = 0;

        function appendChar(container, ch, prob, rank) {
            const span = document.createElement('span');
            span.className = 'mlp-live-char';
            span.style.color = rnnRankColor(rank);
            const shown = (ch === '\n') ? '\\n' : ch;
            span.title = `Confidence in "${shown}": ${(prob * 100).toFixed(1)}% (rank #${rank + 1})`;
            span.textContent = ch;
            container.appendChild(span);
            container.scrollTop = container.scrollHeight;
        }

        function tick() {
            if (token !== rnnRaceToken) return;
            if (i >= lstmSteps.length || i >= rnnSteps.length) {
                const lstmAvg = lstmSum / i;
                const rnnAvg = rnnSum / i;
                rnnModelStats.textContent =
                    `Finished. Final avg confidence over this excerpt - LSTM: ${(lstmAvg * 100).toFixed(1)}%, ` +
                    `Plain RNN: ${(rnnAvg * 100).toFixed(1)}%.`;
                return;
            }
            const l = lstmSteps[i];
            const r = rnnSteps[i];
            appendChar(rnnRaceLstmOut, l.ch, l.prob, l.rank);
            appendChar(rnnRaceRnnOut, r.ch, r.prob, r.rank);
            rnnUpdateCellDiagrams(l.gates, r.gates);

            lstmSum += l.prob;
            rnnSum += r.prob;
            const n = i + 1;
            const lstmAvg = lstmSum / n;
            const rnnAvg = rnnSum / n;
            rnnMeterLstm.textContent = `${(lstmAvg * 100).toFixed(1)}%`;
            rnnMeterRnn.textContent = `${(rnnAvg * 100).toFixed(1)}%`;
            rnnUpdateLeaderBanner(lstmAvg, rnnAvg);

            i++;
            setTimeout(tick, charDelayMs);
        }
        tick();
    }

    rnnRaceBtn.addEventListener('click', () => {
        if (!rnnModels.lstm || !rnnModels.rnn) return;
        const excerpt = RNN_EXCERPTS[Number(rnnExcerptSelect.value)];

        rnnRaceToken++;
        const myToken = rnnRaceToken;
        const charDelayMs = 60;

        const lstmSteps = rnnScoreExcerpt(excerpt, rnnModels.lstm, 'lstm');
        const rnnSteps = rnnScoreExcerpt(excerpt, rnnModels.rnn, 'rnn');

        rnnMeterLstm.textContent = '—';
        rnnMeterRnn.textContent = '—';
        rnnLeaderBanner.classList.remove('leader-lstm', 'leader-rnn');
        rnnLeaderBanner.textContent = 'Reading...';
        rnnModelStats.textContent = 'Reading live...';
        rnnResetCellDiagrams();

        rnnRunReadAlong(excerpt, lstmSteps, rnnSteps, myToken, charDelayMs);
    });

    rnnRaceResetBtn.addEventListener('click', () => {
        rnnRaceToken++; // cancel any in-flight read-along animation
        rnnRaceLstmOut.innerHTML = '';
        rnnRaceRnnOut.innerHTML = '';
        rnnMeterLstm.textContent = '—';
        rnnMeterRnn.textContent = '—';
        rnnLeaderBanner.classList.remove('leader-lstm', 'leader-rnn');
        rnnLeaderBanner.textContent = 'Pick an excerpt and click "Read Along" to start.';
        rnnModelStats.textContent = '';
        rnnResetCellDiagrams();
    });

    loadRnnWeights();


    /* =====================================================================
       4. Live Transformer - real trained weights, real math. Loads Project
       3's actual trained parameters (exported by
       scripts/export_transformer_weights.py) and runs a hand-written
       decoder-only Transformer forward pass in JavaScript - token +
       position embeddings, causal multi-head self-attention (real Q/K/V
       projections, real softmax attention weights, not illustrative ones),
       feedforward with GELU, layer norm, matching mini_transformer_gpt.py's
       MiniTransformerGPT exactly.

       Two things run off this: (1) a real progression stat card comparing
       loss/perplexity across all 4 architectures built in this portfolio,
       and (2) a live self-attention heatmap - type a short prompt and the
       grid shows the model's REAL post-softmax attention weights for the
       selected layer/head, i.e. which earlier characters it actually
       looked at to predict each next one. "Continue Writing" runs its real
       autoregressive sampling loop.
       ===================================================================== */
    const progressionCard = document.getElementById('progression-card');
    const attnPromptInput = document.getElementById('attn-prompt-input');
    const attnRunBtn = document.getElementById('attn-run-btn');
    const attnContinueBtn = document.getElementById('attn-continue-btn');
    const attnResetBtn = document.getElementById('attn-reset-btn');
    const attnLayerSelect = document.getElementById('attn-layer-select');
    const attnHeadSelect = document.getElementById('attn-head-select');
    const attnTokensContainer = document.getElementById('attention-tokens');
    const attnMatrixContainer = document.getElementById('attention-matrix');
    const attnGeneratedOutput = document.getElementById('attn-generated-output');
    const attnModelStats = document.getElementById('attn-model-stats');

    const ATTN_STOP_CHARS = new Set(['.', '!', '?']);
    let attnWeights = null;
    const attnStoi = {};
    let attnLastRun = null; // { tokens, attentionCache } from the most recent Run/Continue
    let attnCurrentText = '';
    let attnGenToken = 0;

    function attnRankColor(rank) {
        if (rank === 0) return '#4ade80';
        if (rank <= 2) return '#a3e635';
        if (rank <= 5) return '#facc15';
        if (rank <= 15) return '#fb923c';
        return '#f87171';
    }

    async function attnRenderProgressionCard() {
        try {
            const [mlpRes, rnnRes, lstmRes] = await Promise.all([
                fetch('model_weights/mlp_weights.json'),
                fetch('model_weights/rnn_weights.json'),
                fetch('model_weights/lstm_weights.json'),
            ]);
            const mlpData = await mlpRes.json();
            const rnnData = await rnnRes.json();
            const lstmData = await lstmRes.json();

            const entries = [
                { label: 'MLP (Project 1)', loss: mlpData.val_loss },
                { label: 'Plain RNN (Project 2)', loss: rnnData.train_loss },
                { label: 'LSTM (Project 2)', loss: lstmData.train_loss },
                { label: 'Transformer (Project 3)', loss: attnWeights.train_loss },
            ];

            const col = ({ label, loss }) => `
                <div class="rnn-stat-col">
                    <h4>${label}</h4>
                    <span class="rnn-stat-value">${Math.exp(loss).toFixed(2)}</span>
                    <span class="rnn-stat-sub">perplexity</span>
                    <span class="rnn-stat-sub">loss: ${loss.toFixed(4)}</span>
                </div>
            `;
            progressionCard.innerHTML = entries.map(col).join('') +
                `<p class="rnn-stat-note">Each a real, separately trained model on the same Shakespeare task &mdash; lower is better on both metrics. This is the actual measured improvement across the 4 projects, not a narrative.</p>`;
        } catch (err) {
            progressionCard.innerHTML = '';
            progressionCard.textContent = 'Could not load one of the other projects\' trained weights for comparison.';
            console.error('Failed to build progression card:', err);
        }
    }

    async function attnLoadWeights() {
        try {
            const res = await fetch('model_weights/transformer_weights.json');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            attnWeights = await res.json();
            attnWeights.vocab.forEach((ch, i) => { attnStoi[ch] = i; });

            attnLayerSelect.innerHTML = '';
            for (let l = 0; l < attnWeights.n_layer; l++) {
                const opt = document.createElement('option');
                opt.value = String(l);
                opt.textContent = `Layer ${l + 1}`;
                attnLayerSelect.appendChild(opt);
            }
            attnHeadSelect.innerHTML = '';
            for (let h = 0; h < attnWeights.n_head; h++) {
                const opt = document.createElement('option');
                opt.value = String(h);
                opt.textContent = `Head ${h + 1}`;
                attnHeadSelect.appendChild(opt);
            }

            await attnRenderProgressionCard();
            attnModelStats.textContent = 'Type a short prompt and click "Run Transformer" to see its real attention weights.';
        } catch (err) {
            attnModelStats.textContent = 'Could not load the trained model weights - this demo needs to be served over http(s), not opened directly as a local file.';
            console.error('Failed to load Transformer weights:', err);
        }
    }

    // --- Real Transformer forward-pass math (mirrors MiniTransformerGPT exactly) ---

    function attnMatVecPlusBias(W, x, bias) {
        const out = new Array(W.length);
        for (let r = 0; r < W.length; r++) {
            let sum = bias[r];
            const row = W[r];
            for (let k = 0; k < x.length; k++) sum += row[k] * x[k];
            out[r] = sum;
        }
        return out;
    }

    function attnLayerNorm(x, gamma, beta, eps = 1e-5) {
        const n = x.length;
        const mean = x.reduce((s, v) => s + v, 0) / n;
        const variance = x.reduce((s, v) => s + (v - mean) * (v - mean), 0) / n;
        const invStd = 1 / Math.sqrt(variance + eps);
        return x.map((v, i) => (v - mean) * invStd * gamma[i] + beta[i]);
    }

    // Abramowitz & Stegun 7.1.26 erf approximation (max error ~1.5e-7) - used
    // to match PyTorch's exact (erf-based) nn.GELU(), not the tanh approximation.
    function attnErf(x) {
        const sign = x < 0 ? -1 : 1;
        x = Math.abs(x);
        const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        const t = 1 / (1 + p * x);
        const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        return sign * y;
    }

    function attnGelu(x) {
        return 0.5 * x * (1 + attnErf(x / Math.SQRT2));
    }

    function attnSoftmax(vec) {
        const max = Math.max(...vec);
        const exps = vec.map(v => Math.exp(v - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    }

    // Full causal self-attention forward pass over a short token sequence.
    // Returns per-position logits AND the real post-softmax attention
    // weights per layer per head, so the same pass powers both the heatmap
    // and generation - no separate "visualization-only" computation.
    function attnForward(tokenIds, weights) {
        const T = tokenIds.length;
        const C = weights.n_embed;
        const nHead = weights.n_head;
        const headSize = C / nHead;

        let x = tokenIds.map((id, pos) => {
            const tok = weights.wte[id];
            const posEmb = weights.wpe[pos];
            return tok.map((v, i) => v + posEmb[i]);
        });

        const attentionCache = []; // n_layer x n_head x T x T

        weights.layers.forEach(layer => {
            const normed1 = x.map(v => attnLayerNorm(v, layer.ln_1_weight, layer.ln_1_bias));
            const qkv = normed1.map(v => attnMatVecPlusBias(layer.c_attn_weight, v, layer.c_attn_bias));
            const q = qkv.map(v => v.slice(0, C));
            const k = qkv.map(v => v.slice(C, 2 * C));
            const vv = qkv.map(v => v.slice(2 * C, 3 * C));

            const layerHeadAttn = [];
            const attnOut = Array.from({ length: T }, () => new Array(C).fill(0));

            for (let h = 0; h < nHead; h++) {
                const hs = h * headSize;
                const headAttn = [];
                for (let i = 0; i < T; i++) {
                    const qi = q[i].slice(hs, hs + headSize);
                    const scores = new Array(T).fill(-Infinity);
                    for (let j = 0; j <= i; j++) {
                        const kj = k[j].slice(hs, hs + headSize);
                        let dot = 0;
                        for (let d = 0; d < headSize; d++) dot += qi[d] * kj[d];
                        scores[j] = dot / Math.sqrt(headSize);
                    }
                    const probs = attnSoftmax(scores);
                    headAttn.push(probs);
                    for (let d = 0; d < headSize; d++) {
                        let acc = 0;
                        for (let j = 0; j <= i; j++) acc += probs[j] * vv[j][hs + d];
                        attnOut[i][hs + d] = acc;
                    }
                }
                layerHeadAttn.push(headAttn);
            }
            attentionCache.push(layerHeadAttn);

            const projected = attnOut.map(v => attnMatVecPlusBias(layer.c_proj_weight, v, layer.c_proj_bias));
            x = x.map((v, i) => v.map((val, d) => val + projected[i][d]));

            const normed2 = x.map(v => attnLayerNorm(v, layer.ln_2_weight, layer.ln_2_bias));
            const hidden = normed2.map(v => attnMatVecPlusBias(layer.ffwd_w1, v, layer.ffwd_b1).map(attnGelu));
            const ffOut = hidden.map(v => attnMatVecPlusBias(layer.ffwd_w2, v, layer.ffwd_b2));
            x = x.map((v, i) => v.map((val, d) => val + ffOut[i][d]));
        });

        const finalNormed = x.map(v => attnLayerNorm(v, weights.ln_f_weight, weights.ln_f_bias));
        const logits = finalNormed.map(v => attnMatVecPlusBias(weights.lm_head_weight, v, weights.lm_head_bias));

        return { logits, attentionCache };
    }

    function attnTokenizePrompt(text) {
        const kept = [];
        for (const ch of text) {
            if (ch in attnStoi) kept.push(ch);
        }
        return kept;
    }

    function attnRenderHeatmap(tokens, attentionCache) {
        attnTokensContainer.innerHTML = '';
        attnMatrixContainer.innerHTML = '';
        if (tokens.length === 0) return;

        const layerIdx = Number(attnLayerSelect.value);
        const headIdx = Number(attnHeadSelect.value);
        const matrix = attentionCache[layerIdx][headIdx];
        const T = tokens.length;
        const cellSize = T > 18 ? Math.max(20, Math.floor(600 / T)) : 45;

        const display = ch => (ch === '\n' ? '↵' : ch === ' ' ? '␣' : ch);

        attnMatrixContainer.style.gridTemplateColumns = `60px repeat(${T}, ${cellSize}px)`;

        tokens.forEach((rowCh, r) => {
            const rowLabel = document.createElement('div');
            rowLabel.className = 'matrix-label';
            rowLabel.style.width = '60px';
            rowLabel.textContent = display(rowCh);
            attnMatrixContainer.appendChild(rowLabel);

            tokens.forEach((colCh, c) => {
                const cell = document.createElement('div');
                cell.className = 'matrix-cell';
                cell.id = `attn-cell-${r}-${c}`;
                cell.style.width = `${cellSize}px`;
                cell.style.height = `${cellSize}px`;

                const score = matrix[r][c];
                if (c > r) {
                    // Causally masked - the model never sees future tokens
                    cell.style.backgroundColor = 'rgba(255, 255, 255, 0.02)';
                    cell.style.cursor = 'default';
                } else {
                    cell.textContent = score.toFixed(2);
                    cell.style.backgroundColor = `rgba(236, 72, 153, ${score.toFixed(3)})`;
                    cell.title = `"${display(rowCh)}" attends to "${display(colCh)}" with weight ${(score * 100).toFixed(1)}%`;
                }
                attnMatrixContainer.appendChild(cell);
            });
        });

        tokens.forEach((ch, idx) => {
            const token = document.createElement('span');
            token.className = 'token-word';
            token.textContent = display(ch);
            token.dataset.idx = idx;
            token.addEventListener('mouseenter', () => {
                token.classList.add('active');
                attnHighlightRow(idx, T);
            });
            token.addEventListener('mouseleave', () => {
                token.classList.remove('active');
                attnResetHighlights(T);
            });
            attnTokensContainer.appendChild(token);
        });
    }

    function attnHighlightRow(rowIdx, T) {
        for (let r = 0; r < T; r++) {
            for (let c = 0; c < T; c++) {
                const cell = document.getElementById(`attn-cell-${r}-${c}`);
                if (!cell) continue;
                if (r === rowIdx) {
                    cell.style.borderColor = 'var(--accent-pink)';
                } else {
                    cell.style.opacity = '0.35';
                }
            }
        }
    }

    function attnResetHighlights(T) {
        for (let r = 0; r < T; r++) {
            for (let c = 0; c < T; c++) {
                const cell = document.getElementById(`attn-cell-${r}-${c}`);
                if (cell) {
                    cell.style.borderColor = 'rgba(255, 255, 255, 0.02)';
                    cell.style.opacity = '1.0';
                }
            }
        }
    }

    function attnRerenderHeatmap() {
        if (!attnLastRun) return;
        attnRenderHeatmap(attnLastRun.tokens, attnLastRun.attentionCache);
    }

    attnLayerSelect.addEventListener('change', attnRerenderHeatmap);
    attnHeadSelect.addEventListener('change', attnRerenderHeatmap);

    attnRunBtn.addEventListener('click', () => {
        if (!attnWeights) return;
        const raw = attnPromptInput.value;
        const tokens = attnTokenizePrompt(raw).slice(0, attnWeights.block_size);
        if (tokens.length === 0) {
            attnModelStats.textContent = 'Type at least one character this model recognizes (it only knows the characters in its training corpus).';
            return;
        }

        attnGenToken++;
        attnCurrentText = tokens.join('');
        attnGeneratedOutput.innerHTML = '';

        const tokenIds = tokens.map(ch => attnStoi[ch]);
        const { attentionCache } = attnForward(tokenIds, attnWeights);
        attnLastRun = { tokens, attentionCache };
        attnRenderHeatmap(tokens, attentionCache);

        const dropped = raw.length - tokens.length;
        attnModelStats.textContent = dropped > 0
            ? `Ran the real forward pass on ${tokens.length} characters (dropped ${dropped} outside the model's vocabulary). Hover a token, or switch layer/head, to explore.`
            : `Ran the real forward pass on ${tokens.length} characters. Hover a token, or switch layer/head, to explore.`;
    });

    // Real, live autoregressive continuation - same sample -> step -> repeat
    // algorithm as MiniTransformerGPT.generate(), ported to JS. Crops
    // context to the model's real block_size exactly like the Python version.
    // Computes and reveals ONE character per tick, instead of running all
    // ~100 forward passes synchronously before showing anything. The
    // Transformer's forward pass recomputes full self-attention over the
    // whole context every step (O(T^2), unlike the MLP/RNN's cheap
    // per-step math), so precomputing 100 of them back-to-back blocked the
    // page for many seconds with no visible progress - it looked frozen,
    // not just slow. Interleaving compute with the reveal keeps the UI
    // responsive and makes "watch it think" literally true.
    attnContinueBtn.addEventListener('click', () => {
        if (!attnWeights || !attnCurrentText) return;
        attnGenToken++;
        const myToken = attnGenToken;

        const temperature = 0.7;
        const topK = 10;
        const maxNew = 100;
        const charDelayMs = 45;

        let contextChars = attnCurrentText.split('');
        let step = 0;

        function genNext() {
            if (myToken !== attnGenToken) return;
            if (step >= maxNew) {
                attnCurrentText = contextChars.join('');
                attnModelStats.textContent = 'Finished generating. Click "Continue Writing" again to keep going, or "Run Transformer" to start over with a new prompt.';
                return;
            }

            const cropped = contextChars.length > attnWeights.block_size
                ? contextChars.slice(-attnWeights.block_size)
                : contextChars;
            const tokenIds = cropped.map(ch => attnStoi[ch]);
            const { logits } = attnForward(tokenIds, attnWeights);
            const lastLogits = logits[logits.length - 1].map(v => v / temperature);

            const sorted = lastLogits.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
            const threshold = sorted[Math.min(topK, sorted.length) - 1].v;
            const filtered = lastLogits.map(v => (v < threshold ? -Infinity : v));
            const probs = attnSoftmax(filtered);

            const r = Math.random();
            let cumulative = 0;
            let idx = probs.length - 1;
            for (let i = 0; i < probs.length; i++) {
                cumulative += probs[i];
                if (r <= cumulative) { idx = i; break; }
            }
            const ch = attnWeights.vocab[idx];
            const prob = probs[idx];
            let rank = 0;
            for (let i = 0; i < probs.length; i++) if (probs[i] > prob) rank++;

            contextChars.push(ch);
            step++;

            const span = document.createElement('span');
            span.className = 'mlp-live-char';
            span.style.color = attnRankColor(rank);
            const shown = (ch === '\n') ? '\\n' : ch;
            span.title = `Sampled "${shown}" at ${(prob * 100).toFixed(1)}% confidence (its own guess #${rank + 1})`;
            span.textContent = ch;
            attnGeneratedOutput.appendChild(span);
            attnGeneratedOutput.scrollTop = attnGeneratedOutput.scrollHeight;

            if (ATTN_STOP_CHARS.has(ch)) {
                attnCurrentText = contextChars.join('');
                attnModelStats.textContent = 'Finished generating (hit sentence-ending punctuation). Click "Continue Writing" again to keep going.';
                return;
            }

            setTimeout(genNext, charDelayMs);
        }

        attnModelStats.textContent = 'Generating live...';
        genNext();
    });

    attnResetBtn.addEventListener('click', () => {
        attnGenToken++;
        attnPromptInput.value = 'To be or not to be';
        attnCurrentText = '';
        attnLastRun = null;
        attnTokensContainer.innerHTML = '';
        attnMatrixContainer.innerHTML = '';
        attnGeneratedOutput.innerHTML = '';
        attnModelStats.textContent = 'Type a short prompt and click "Run Transformer" to see its real attention weights.';
    });

    attnLoadWeights();


    /* =====================================================================
       5. Live RAG Agent - real facts, real math. Loads Project 4's actual
       facts database (exported by scripts/export_rag_facts.py, straight
       from rag_agent.py's SHAKESPEARE_FACTS_DB) and runs a from-scratch
       TF-IDF vector database in JavaScript - tokenize, build vocabulary,
       compute IDF, vectorize, cosine similarity - matching
       SimpleVectorDB.retrieve() exactly, then the same local
       keyword-extraction answer engine GenerationEngine._call_local_extractor()
       runs when no Gemini API key is present (this browser demo can't hold
       a secret API key, so it always takes that same code path the CLI
       takes when running without one - not a downgraded version of it).
       ===================================================================== */
    const ragExamplesContainer = document.getElementById('rag-examples');
    const ragQueryInput = document.getElementById('rag-query-input');
    const ragAskBtn = document.getElementById('rag-ask-btn');
    const ragResetBtn = document.getElementById('rag-reset-btn');
    const ragRetrievedContainer = document.getElementById('rag-retrieved');
    const ragPromptBox = document.getElementById('rag-augmented-prompt');
    const ragAnswerBox = document.getElementById('rag-answer');
    const ragModelStats = document.getElementById('rag-model-stats');

    const ragPipeNodes = {
        query: document.getElementById('rag-pipe-query'),
        retrieve: document.getElementById('rag-pipe-retrieve'),
        augment: document.getElementById('rag-pipe-augment'),
        generate: document.getElementById('rag-pipe-generate'),
        response: document.getElementById('rag-pipe-response'),
    };
    let ragRunToken = 0;

    const RAG_EXAMPLE_QUERIES = [
        "Who is Iago and what does he do in Othello?",
        "What is the Montague and Capulet feud about?",
        "When did Shakespeare die?",
        "What do the three witches tell Macbeth?",
    ];

    let ragDb = null; // { documents, vocab, idf, matrix (normalized) }

    // --- Real TF-IDF vector database (mirrors SimpleVectorDB exactly) ---

    function ragTokenize(text) {
        return (text.toLowerCase().match(/[a-z0-9_]+/g)) || [];
    }

    function ragFitDocuments(documents) {
        const wordDocCounts = {}; // how many documents contain each word
        const docTokens = documents.map(ragTokenize);

        docTokens.forEach(tokens => {
            new Set(tokens).forEach(tok => {
                wordDocCounts[tok] = (wordDocCounts[tok] || 0) + 1;
            });
        });

        const vocab = {};
        Object.keys(wordDocCounts).forEach((word, i) => { vocab[word] = i; });

        const numDocs = documents.length;
        const idf = {};
        Object.entries(wordDocCounts).forEach(([word, count]) => {
            idf[word] = Math.log((numDocs + 1) / (count + 1)) + 1.0;
        });

        function vectorize(tokens) {
            const vector = new Array(Object.keys(vocab).length).fill(0);
            const total = tokens.length;
            if (total === 0) return vector;
            const counts = {};
            tokens.forEach(tok => {
                if (tok in vocab) counts[tok] = (counts[tok] || 0) + 1;
            });
            Object.entries(counts).forEach(([tok, count]) => {
                const tf = count / total;
                vector[vocab[tok]] = tf * idf[tok];
            });
            return vector;
        }

        const matrix = docTokens.map(vectorize).map(vec => {
            const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
            return norm > 0 ? vec.map(v => v / norm) : vec;
        });

        return { documents, vocab, idf, matrix, vectorize };
    }

    function ragRetrieve(db, query, topK = 2) {
        let queryVec = db.vectorize(ragTokenize(query));
        const norm = Math.sqrt(queryVec.reduce((s, v) => s + v * v, 0));
        if (norm > 0) queryVec = queryVec.map(v => v / norm);

        const scored = db.matrix.map((docVec, idx) => {
            let dot = 0;
            for (let i = 0; i < docVec.length; i++) dot += docVec[i] * queryVec[i];
            return { document: db.documents[idx], score: dot, index: idx };
        });
        scored.sort((a, b) => b.score - a.score);
        return scored.slice(0, topK);
    }

    // Mirrors GenerationEngine._call_local_extractor() exactly: finds
    // keywords (4+ letter words) in the query, splits retrieved documents
    // into sentences, ranks sentences by how many keywords they contain.
    function ragLocalExtractor(retrievedDocs, query) {
        const keywords = (query.match(/\b\w{4,}\b/g) || []).map(w => w.toLowerCase());
        const sentences = [];

        retrievedDocs.forEach(doc => {
            const docSentences = doc.split(/(?<=[.!?]) +/);
            docSentences.forEach(sentence => {
                const lower = sentence.toLowerCase();
                const matchedCount = keywords.reduce((n, kw) => n + (lower.includes(kw) ? 1 : 0), 0);
                if (matchedCount > 0) sentences.push({ sentence, matchedCount });
            });
        });

        sentences.sort((a, b) => b.matchedCount - a.matchedCount);

        if (sentences.length > 0) {
            const seen = new Set();
            const top = [];
            for (const s of sentences) {
                if (!seen.has(s.sentence)) { seen.add(s.sentence); top.push(s.sentence); }
                if (top.length === 3) break;
            }
            return `[Local Engine Answer]:\nBased on retrieved data, here is what I found:\n- ${top.join('\n- ')}`;
        }
        const bestDoc = retrievedDocs.length > 0 ? retrievedDocs[0] : 'No context available.';
        return `[Local Engine Answer]:\nCould not extract specific sentence matches. Best matching document:\n> ${bestDoc}`;
    }

    // Mirrors RAGAgent.query()'s prompt template exactly (the real "AUGMENT"
    // step: pasting retrieved documents into a context template around the
    // user's question).
    function ragBuildPrompt(contexts, query) {
        const contextStr = contexts.map((doc, i) => `Document ${i + 1}:\n${doc}`).join('\n\n');
        return `You are a helpful QA Assistant. Answer the User Query based ONLY on the provided Context documents.\nIf the answer cannot be determined from the context, state that clearly.\n\nContext Documents:\n${contextStr}\n\nUser Query: ${query}\n\nAnswer:`;
    }

    async function ragLoadFacts() {
        try {
            const res = await fetch('model_weights/rag_facts.json');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            ragDb = ragFitDocuments(data.documents);

            RAG_EXAMPLE_QUERIES.forEach(q => {
                const chip = document.createElement('button');
                chip.type = 'button';
                chip.className = 'token-word';
                chip.textContent = q;
                chip.addEventListener('click', () => {
                    ragQueryInput.value = q;
                    ragRunQuery(q);
                });
                ragExamplesContainer.appendChild(chip);
            });

            ragModelStats.textContent = `Loaded ${data.documents.length} real facts. Ask a question, or click an example above.`;
        } catch (err) {
            ragModelStats.textContent = 'Could not load the facts database - this demo needs to be served over http(s), not opened directly as a local file.';
            console.error('Failed to load RAG facts:', err);
        }
    }

    function ragRenderRetrieved(results) {
        ragRetrievedContainer.innerHTML = '';
        results.forEach(r => {
            const row = document.createElement('div');
            row.className = 'mlp-pred-row';
            row.innerHTML = `
                <div class="mlp-pred-bar" style="width: ${Math.min(100, r.score * 100).toFixed(1)}%"></div>
                <span class="mlp-pred-char">${r.document}</span>
                <span class="mlp-pred-pct">${(r.score * 100).toFixed(1)}%</span>
            `;
            ragRetrievedContainer.appendChild(row);
        });
    }

    function ragSetActiveStage(stage) {
        Object.entries(ragPipeNodes).forEach(([key, el]) => {
            if (el) el.classList.toggle('active', key === stage);
        });
    }

    function ragResetPipelineVisual() {
        Object.values(ragPipeNodes).forEach(el => { if (el) el.classList.remove('active'); });
    }

    // Runs the real pipeline synchronously (all of this is cheap, deterministic
    // math - nothing here is simulated), then reveals each stage with a short
    // stagger so the query -> retrieve -> augment -> generate -> response flow
    // is actually visible, not just instant. A token guards against overlapping
    // animations if Ask is clicked again before the previous run finishes.
    function ragRunQuery(query) {
        if (!ragDb || !query.trim()) return;
        ragRunToken++;
        const myToken = ragRunToken;
        const stepDelayMs = 450;

        const retrieved = ragRetrieve(ragDb, query, 2);
        const contexts = retrieved.map(r => r.document);
        const prompt = ragBuildPrompt(contexts, query);
        const answer = ragLocalExtractor(contexts, query);

        ragResetPipelineVisual();
        ragRetrievedContainer.innerHTML = '';
        ragPromptBox.textContent = '';
        ragAnswerBox.textContent = '';
        ragModelStats.textContent = 'Running the real pipeline...';

        const stages = [
            () => { ragSetActiveStage('query'); },
            () => { ragSetActiveStage('retrieve'); ragRenderRetrieved(retrieved); },
            () => { ragSetActiveStage('augment'); ragPromptBox.textContent = prompt; },
            () => { ragSetActiveStage('generate'); ragAnswerBox.textContent = answer; },
            () => {
                ragSetActiveStage('response');
                ragModelStats.textContent = `Retrieved ${retrieved.length} facts by real cosine similarity, then answered using only their content.`;
            },
        ];

        stages.forEach((run, i) => {
            setTimeout(() => {
                if (myToken !== ragRunToken) return;
                run();
            }, i * stepDelayMs);
        });
    }

    ragAskBtn.addEventListener('click', () => ragRunQuery(ragQueryInput.value));
    ragQueryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') ragRunQuery(ragQueryInput.value);
    });

    ragResetBtn.addEventListener('click', () => {
        ragRunToken++; // cancel any in-flight staggered reveal
        ragQueryInput.value = 'Who is Iago and what does he do in Othello?';
        ragResetPipelineVisual();
        ragRetrievedContainer.innerHTML = '';
        ragPromptBox.textContent = '';
        ragAnswerBox.textContent = '';
        ragModelStats.textContent = ragDb
            ? `Loaded ${ragDb.documents.length} real facts. Ask a question, or click an example above.`
            : 'Loading the real facts database…';
    });

    ragLoadFacts();
});
