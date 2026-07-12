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
            } else if (targetTab === 'attention-tab') {
                initAttentionMatrix();
            }
        });
    });

    /* =====================================================================
       2. Neural Network Forward Visualizer
       ===================================================================== */
    const mlpSvg = document.getElementById('mlp-svg');
    const mlpForwardBtn = document.getElementById('mlp-forward-btn');
    const mlpResetBtn = document.getElementById('mlp-reset-btn');
    const mlpOut0 = document.getElementById('mlp-out-0');
    const mlpOut1 = document.getElementById('mlp-out-1');

    // Layer specifications: Input (2), Hidden (3), Output (2)
    const layerSizes = [2, 3, 2];
    const layerPositionsX = [80, 250, 420]; // X coordinates for layers
    let nodeElements = [];
    let edgeElements = [];

    function initMlpSvg() {
        mlpSvg.innerHTML = ''; // Clear SVG
        nodeElements = [];
        edgeElements = [];

        const svgWidth = 500;
        const svgHeight = 300;

        // 1. Draw Edges (Weights connection)
        for (let l = 0; l < layerSizes.length - 1; l++) {
            const currentLayerSize = layerSizes[l];
            const nextLayerSize = layerSizes[l+1];
            
            const x1 = layerPositionsX[l];
            const x2 = layerPositionsX[l+1];
            
            for (let i = 0; i < currentLayerSize; i++) {
                // Y coordinates offset to vertically center nodes
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
                    edgeElements.push(line);
                }
            }
        }

        // 2. Draw Nodes
        for (let l = 0; l < layerSizes.length; l++) {
            const layerSize = layerSizes[l];
            const x = layerPositionsX[l];
            
            for (let i = 0; i < layerSize; i++) {
                const y = (svgHeight / (layerSize + 1)) * (i + 1);
                
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', x);
                circle.setAttribute('cy', y);
                circle.setAttribute('r', 16);
                circle.setAttribute('class', 'node');
                circle.id = `node-${l}-${i}`;
                
                // Add labels inside nodes
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x);
                text.setAttribute('y', y + 5);
                text.setAttribute('text-anchor', 'middle');
                text.setAttribute('fill', '#94a3b8');
                text.setAttribute('font-size', '10px');
                text.setAttribute('font-family', 'sans-serif');
                
                let label = '';
                if (l === 0) label = `X${i+1}`;
                else if (l === 1) label = `H${i+1}`;
                else label = `Y${i+1}`;
                
                text.textContent = label;
                
                mlpSvg.appendChild(circle);
                mlpSvg.appendChild(text);
                nodeElements.push(circle);
            }
        }
    }

    // Trigger sequential activation sweeps
    mlpForwardBtn.addEventListener('click', () => {
        // Reset states
        nodeElements.forEach(n => n.classList.remove('active'));
        edgeElements.forEach(e => e.classList.remove('active'));
        mlpOut0.textContent = '0.00';
        mlpOut1.textContent = '0.00';

        // 1. Activate Input Layer (Layer 0)
        setTimeout(() => {
            document.querySelectorAll('#node-0-0, #node-0-1').forEach(node => node.classList.add('active'));
        }, 100);

        // 2. Activate weights Layer 0 -> Layer 1
        setTimeout(() => {
            document.querySelectorAll('[id^="edge-0-"]').forEach(edge => edge.classList.add('active'));
        }, 600);

        // 3. Activate Hidden Layer (Layer 1)
        setTimeout(() => {
            document.querySelectorAll('#node-1-0, #node-1-1, #node-1-2').forEach(node => node.classList.add('active'));
        }, 1100);

        // 4. Activate weights Layer 1 -> Layer 2
        setTimeout(() => {
            document.querySelectorAll('[id^="edge-1-"]').forEach(edge => edge.classList.add('active'));
        }, 1600);

        // 5. Activate Output Layer (Layer 2) and display classification scores
        setTimeout(() => {
            document.querySelectorAll('#node-2-0, #node-2-1').forEach(node => node.classList.add('active'));
            // Mock softmax calculations
            const r1 = (Math.random() * 0.2 + 0.75).toFixed(2);
            const r2 = (1.0 - parseFloat(r1)).toFixed(2);
            mlpOut0.textContent = r1;
            mlpOut1.textContent = r2;
        }, 2100);
    });

    mlpResetBtn.addEventListener('click', () => {
        nodeElements.forEach(n => n.classList.remove('active'));
        edgeElements.forEach(e => e.classList.remove('active'));
        mlpOut0.textContent = '0.00';
        mlpOut1.textContent = '0.00';
    });


    /* =====================================================================
       3. RNN Sequential State Unroller
       ===================================================================== */
    const rnnInputWord = document.getElementById('rnn-input-word');
    const rnnStepBtn = document.getElementById('rnn-step-btn');
    const rnnSeqStrip = document.getElementById('rnn-seq-strip');
    const rnnHiddenCells = document.getElementById('rnn-hidden-cells');

    rnnStepBtn.addEventListener('click', () => {
        const word = rnnInputWord.value.trim();
        if (!word) return;

        // Clear previous list
        rnnSeqStrip.innerHTML = '';

        const chars = word.split('');
        let currentIdx = 0;

        function processNextChar() {
            if (currentIdx >= chars.length) return;

            const char = chars[currentIdx];

            // 1. Create character block element
            const node = document.createElement('div');
            node.className = 'seq-node processed';
            node.innerHTML = `
                <span class="char">${char.toUpperCase()}</span>
                <span class="label">x<sub>${currentIdx+1}</sub></span>
            `;
            rnnSeqStrip.appendChild(node);

            // 2. Add connecting arrow if there's a following character
            if (currentIdx < chars.length - 1) {
                const arrow = document.createElement('div');
                arrow.className = 'seq-arrow';
                arrow.textContent = '➔';
                rnnSeqStrip.appendChild(arrow);
            }

            // Scroll container to keep tracking items
            rnnSeqStrip.scrollLeft = rnnSeqStrip.scrollWidth;

            // 3. Randomize hidden vector values to simulate memory state updates
            const hiddenStates = [];
            for (let i = 0; i < 4; i++) {
                // Values between -1.0 and 1.0 (standard tanh range)
                const val = (Math.random() * 2 - 1).toFixed(2);
                hiddenStates.push(val);
            }

            const cells = rnnHiddenCells.querySelectorAll('.vector-cell');
            cells.forEach((cell, idx) => {
                cell.textContent = hiddenStates[idx];
                cell.classList.add('active');
                setTimeout(() => cell.classList.remove('active'), 250);
            });

            currentIdx++;
            
            // Trigger next loop step
            setTimeout(processNextChar, 700);
        }

        processNextChar();
    });


    /* =====================================================================
       4. Self-Attention Matrix Visualizer
       ===================================================================== */
    const attnMatrixContainer = document.getElementById('attention-matrix');
    const attnTokensContainer = document.getElementById('attention-tokens');
    const attnHeadSelect = document.getElementById('attn-head-select');

    const sentence = ["Attention", "Is", "All", "You", "Need"];
    
    // Head weights matching selected modes
    const headWeights = {
        // Head 1: Syntactic Focus (highlights diagonal and neighboring words)
        0: [
            [0.6, 0.3, 0.1, 0.0, 0.0],
            [0.2, 0.5, 0.2, 0.1, 0.0],
            [0.1, 0.2, 0.5, 0.2, 0.0],
            [0.0, 0.1, 0.2, 0.5, 0.2],
            [0.0, 0.0, 0.1, 0.3, 0.6]
        ],
        // Head 2: Contextual Links (connects specific word pairs)
        1: [
            [0.2, 0.1, 0.1, 0.1, 0.5], // Attention attends heavily to Need
            [0.1, 0.4, 0.3, 0.1, 0.1], 
            [0.1, 0.1, 0.5, 0.2, 0.1], 
            [0.5, 0.1, 0.1, 0.2, 0.1], // You attends to Attention
            [0.3, 0.1, 0.1, 0.1, 0.4]  
        ],
        // Head 3: Long-range Anchors (focuses heavily on the last token "Need")
        2: [
            [0.1, 0.1, 0.1, 0.1, 0.6],
            [0.1, 0.1, 0.1, 0.1, 0.6],
            [0.1, 0.1, 0.1, 0.1, 0.6],
            [0.1, 0.1, 0.1, 0.1, 0.6],
            [0.1, 0.1, 0.1, 0.1, 0.6]
        ]
    };

    function initAttentionMatrix() {
        attnMatrixContainer.innerHTML = '';
        attnTokensContainer.innerHTML = '';
        
        const currentHead = parseInt(attnHeadSelect.value);
        const matrixData = headWeights[currentHead];
        
        // 1. Set grid column dimensions dynamically
        attnMatrixContainer.style.gridTemplateColumns = `80px repeat(${sentence.length}, 45px)`;

        // 2. Generate Grid
        sentence.forEach((rowWord, rIdx) => {
            // Add row label
            const rowLabel = document.createElement('div');
            rowLabel.className = 'matrix-label';
            rowLabel.textContent = rowWord;
            attnMatrixContainer.appendChild(rowLabel);

            // Add grid cells
            sentence.forEach((colWord, cIdx) => {
                const cell = document.createElement('div');
                cell.className = 'matrix-cell';
                cell.id = `cell-${rIdx}-${cIdx}`;
                
                const score = matrixData[rIdx][cIdx];
                cell.textContent = score.toFixed(1);
                
                // Scale pink background transparency based on attention score (0.0 to 1.0)
                cell.style.backgroundColor = `rgba(236, 72, 153, ${score})`;
                
                attnMatrixContainer.appendChild(cell);
            });
        });

        // 3. Render sentence tokens row
        sentence.forEach((word, idx) => {
            const token = document.createElement('span');
            token.className = 'token-word';
            token.textContent = word;
            token.dataset.idx = idx;
            
            // Hover events to highlight corresponding matrix rows
            token.addEventListener('mouseenter', () => {
                token.classList.add('active');
                highlightMatrixRow(idx);
            });
            
            token.addEventListener('mouseleave', () => {
                token.classList.remove('active');
                resetMatrixHighlights();
            });

            attnTokensContainer.appendChild(token);
        });
    }

    function highlightMatrixRow(rowIdx) {
        // Light up all cells in target row and dim out other rows
        for (let r = 0; r < sentence.length; r++) {
            for (let c = 0; c < sentence.length; c++) {
                const cell = document.getElementById(`cell-${r}-${c}`);
                if (!cell) continue;
                
                if (r === rowIdx) {
                    cell.style.borderColor = 'var(--accent-pink)';
                    cell.style.color = '#fff';
                } else {
                    cell.style.opacity = '0.25';
                }
            }
        }
    }

    function resetMatrixHighlights() {
        for (let r = 0; r < sentence.length; r++) {
            for (let c = 0; c < sentence.length; c++) {
                const cell = document.getElementById(`cell-${r}-${c}`);
                if (cell) {
                    cell.style.borderColor = 'rgba(255, 255, 255, 0.02)';
                    cell.style.color = 'transparent';
                    cell.style.opacity = '1.0';
                }
            }
        }
    }

    attnHeadSelect.addEventListener('change', initAttentionMatrix);


    /* =====================================================================
       5. Autoregressive Generator Terminal Simulator
       ===================================================================== */
    const tempSlider = document.getElementById('temp-slider');
    const tempVal = document.getElementById('temp-val');
    const tokensSlider = document.getElementById('tokens-slider');
    const tokensVal = document.getElementById('tokens-val');
    const triggerGenBtn = document.getElementById('trigger-gen-btn');
    const generatorTerminal = document.getElementById('generator-terminal');

    // Update slider label displays
    tempSlider.addEventListener('input', (e) => {
        tempVal.textContent = parseFloat(e.target.value).toFixed(1);
    });

    tokensSlider.addEventListener('input', (e) => {
        tokensVal.textContent = e.target.value;
    });

    const standardCodeSnippet = `
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, n_embed=128):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.attention = CausalSelfAttention(n_embed, n_head=4)
        self.ffn = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx):
        x = self.token_embeddings(idx)
        x = x + self.attention(x)
        x = x + self.ffn(x)
        return self.lm_head(x)
`.trim();

    // Glitchy tokens to insert when temperature slider is set high
    const randomGlitches = [
        " # WHat is going on here?? ",
        " # entropy_overflow_warning ",
        " def glitchy_attention_calc(q, k, v): ",
        " # TODO: fix gradients tomorrow... ",
        " print('NaN detected! Continuing anyway...') ",
        " import numpy_as_pytorch_fallback_maybe "
    ];

    triggerGenBtn.addEventListener('click', () => {
        triggerGenBtn.disabled = true;
        const temp = parseFloat(tempSlider.value);
        const maxTokens = parseInt(tokensSlider.value);

        // Reset Terminal
        generatorTerminal.innerHTML = '<span class="terminal-prompt">guest@shabih-ai:~$</span> running generator.py --temp=' + temp + ' --max_tokens=' + maxTokens + '<br><br>';

        let textToType = standardCodeSnippet;
        
        // Mutate string if temperature is high (> 1.2)
        if (temp > 1.2) {
            const lines = standardCodeSnippet.split('\n');
            // Insert random glitchy lines
            for (let i = 0; i < 3; i++) {
                const randomLineIdx = Math.floor(Math.random() * lines.length);
                const randomGlitch = randomGlitches[Math.floor(Math.random() * randomGlitches.length)];
                lines.splice(randomLineIdx, 0, randomGlitch);
            }
            textToType = lines.join('\n');
        }

        // Truncate based on tokens slider
        const charLimit = Math.min(textToType.length, maxTokens * 5); // approximate chars
        textToType = textToType.substring(0, charLimit);

        let charIdx = 0;
        
        function typeChar() {
            if (charIdx >= textToType.length) {
                generatorTerminal.innerHTML += '<br><br><span class="terminal-prompt">guest@shabih-ai:~$</span> <span class="terminal-cursor">|</span>';
                triggerGenBtn.disabled = false;
                return;
            }

            const char = textToType[charIdx];
            
            // Format newlines correctly for HTML printing
            if (char === '\n') {
                generatorTerminal.innerHTML += '<br>';
            } else if (char === ' ') {
                generatorTerminal.innerHTML += '&nbsp;';
            } else {
                // Simple escaping for code tags
                if (char === '<') generatorTerminal.innerHTML += '&lt;';
                else if (char === '>') generatorTerminal.innerHTML += '&gt;';
                else generatorTerminal.innerHTML += char;
            }

            // Scroll terminal to keep typing visible
            generatorTerminal.scrollTop = generatorTerminal.scrollHeight;
            charIdx++;
            
            // Speed of typing
            setTimeout(typeChar, 12);
        }

        typeChar();
    });

    // Default initialization
    initMlpSvg();
    initAttentionMatrix();
});
