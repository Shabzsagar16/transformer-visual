/**
 * Transformer Visualizer - Interactive JavaScript
 * Simulates data flow through a Transformer model with shape tracking
 */

// Toggle code panel visibility
function toggleCode(panelId) {
    const panel = document.getElementById(panelId);
    if (panel) {
        panel.classList.toggle('active');
    }
}

// Scroll to a phase
function scrollToPhase(phaseId) {
    const phase = document.getElementById(phaseId);
    if (phase) {
        phase.scrollIntoView({ behavior: 'smooth', block: 'start' });
        // Highlight the phase briefly
        phase.style.boxShadow = '0 0 20px rgba(34, 197, 94, 0.5)';
        setTimeout(() => {
            phase.style.boxShadow = '';
        }, 1500);
    }
}

// Configuration
const CONFIG = {
    d_model: 512,
    h: 8,
    d_k: 64,
    d_ff: 2048,
    N: 6,
    vocab_size: 10000
};

// Simple word-to-id mapping (simulated tokenizer)
const VOCAB = {
    '[SOS]': 1,
    '[EOS]': 2,
    '[PAD]': 0,
    'the': 45,
    'cat': 892,
    'sat': 234,
    'on': 156,
    'mat': 567,
    'dog': 891,
    'runs': 445,
    'fast': 223,
    'a': 12,
    'is': 78,
    'big': 334,
    'small': 556,
    'happy': 789,
    'jumped': 901,
    'over': 123,
    'lazy': 456,
    'fox': 678,
    'hello': 111,
    'world': 222,
    'i': 33,
    'love': 444,
    'transformers': 999,
    'attention': 888,
    'all': 777,
    'you': 666,
    'need': 555
};

// DOM Elements
const inputSentence = document.getElementById('inputSentence');
const processBtn = document.getElementById('processBtn');

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    processBtn.addEventListener('click', processSentence);
    // Process default sentence
    processSentence();
});

/**
 * Main processing function
 */
function processSentence() {
    const sentence = inputSentence.value.trim().toLowerCase();
    if (!sentence) return;

    // Phase 1: Tokenization
    const tokens = tokenize(sentence);
    displayTokenization(sentence, tokens);

    // Phase 2: Embedding
    const embeddings = generateEmbeddings(tokens);
    displayEmbedding(tokens, embeddings);

    // Phase 3: Positional Encoding
    displayPositionalEncoding(tokens, embeddings);

    // Phase 4: Encoder
    displayEncoder(tokens);

    // Phase 5: Decoder
    const targetTokens = ['[SOS]', 'il', 'gatto', '...'];
    displayDecoder(tokens, targetTokens);

    // Phase 6: Output
    displayOutput(tokens, targetTokens);
}

/**
 * Tokenize input sentence
 */
function tokenize(sentence) {
    const words = sentence.split(/\s+/);
    const tokens = ['[SOS]', ...words, '[EOS]'];
    return tokens;
}

/**
 * Display Phase 1: Tokenization
 */
function displayTokenization(sentence, tokens) {
    // Input text
    document.getElementById('inputText').textContent = `"${sentence}"`;

    // Tokens
    const tokensBox = document.getElementById('tokens');
    tokensBox.innerHTML = tokens.map(t =>
        `<span class="token ${t.startsWith('[') ? 'special' : ''}">${t}</span>`
    ).join('');

    // Tokens shape
    const tokensShape = document.getElementById('tokensShape');
    if (tokensShape) tokensShape.textContent = `(${tokens.length} tokens)`;

    // Token IDs
    const idsBox = document.getElementById('tokenIds');
    idsBox.innerHTML = tokens.map(t => {
        const id = VOCAB[t.toLowerCase()] || Math.floor(Math.random() * 1000) + 100;
        return `<span class="token-id ${t.startsWith('[') ? 'special' : ''}">${id}</span>`;
    }).join('');

    // Shape badge
    document.getElementById('tokenShape').textContent = `Shape: (${tokens.length},)`;
}

/**
 * Generate simulated embeddings
 */
function generateEmbeddings(tokens) {
    return tokens.map(() => {
        const vec = [];
        for (let i = 0; i < 8; i++) { // Show first 8 dims
            vec.push((Math.random() * 2 - 1).toFixed(2));
        }
        return vec;
    });
}

/**
 * Display Phase 2: Embedding
 */
function displayEmbedding(tokens, embeddings) {
    const seqLen = tokens.length;

    // Update shapes
    const embInputShape = document.getElementById('embInputShape');
    if (embInputShape) embInputShape.textContent = `(${seqLen},)`;

    const embOutputShape = document.getElementById('embOutputShape');
    if (embOutputShape) embOutputShape.textContent = `(${seqLen}, 512)`;

    // Matrix display
    const matrix = document.getElementById('embeddingMatrix');
    matrix.innerHTML = embeddings.map((row, i) => `
        <div class="matrix-row">
            <span class="matrix-cell highlight">${truncate(tokens[i], 6)}</span>
            ${row.map(v => `<span class="matrix-cell">${v}</span>`).join('')}
            <span class="matrix-cell">...</span>
        </div>
    `).join('');
}

/**
 * Display Phase 3: Positional Encoding
 */
function displayPositionalEncoding(tokens, embeddings) {
    const seqLen = tokens.length;

    // Update shapes
    const peInputShape1 = document.getElementById('peInputShape1');
    if (peInputShape1) peInputShape1.textContent = `(${seqLen}, 512)`;

    const peInputShape2 = document.getElementById('peInputShape2');
    if (peInputShape2) peInputShape2.textContent = `(${seqLen}, 512)`;

    const peOutputShape = document.getElementById('peOutputShape');
    if (peOutputShape) peOutputShape.textContent = `(${seqLen}, 512)`;

    // Embedding input
    const embInput = document.getElementById('embInput');
    embInput.innerHTML = embeddings.slice(0, 4).map((row, i) => `
        <div class="matrix-row">
            <span class="matrix-cell">${row.slice(0, 3).join(' ')}</span>
            <span class="matrix-cell">...</span>
        </div>
    `).join('') + '<div class="matrix-row"><span class="matrix-cell">⋮</span></div>';

    // PE matrix
    const peMatrix = document.getElementById('peMatrix');
    peMatrix.innerHTML = generatePEVisualization(seqLen);

    // Combined
    const embPlusPos = document.getElementById('embPlusPos');
    embPlusPos.innerHTML = embeddings.slice(0, 4).map((row, i) => `
        <div class="matrix-row">
            <span class="matrix-cell">${row.slice(0, 3).map(v =>
        (parseFloat(v) + Math.sin(i * 0.1)).toFixed(2)
    ).join(' ')}</span>
            <span class="matrix-cell">...</span>
        </div>
    `).join('') + '<div class="matrix-row"><span class="matrix-cell">⋮</span></div>';
}

/**
 * Generate PE visualization
 */
function generatePEVisualization(seqLen) {
    let html = '';
    for (let pos = 0; pos < Math.min(4, seqLen); pos++) {
        html += `<div class="matrix-row">
            <span class="matrix-cell">${Math.sin(pos / Math.pow(10000, 0)).toFixed(2)}</span>
            <span class="matrix-cell">${Math.cos(pos / Math.pow(10000, 0)).toFixed(2)}</span>
            <span class="matrix-cell">${Math.sin(pos / Math.pow(10000, 2 / CONFIG.d_model)).toFixed(2)}</span>
            <span class="matrix-cell">...</span>
        </div>`;
    }
    html += '<div class="matrix-row"><span class="matrix-cell">⋮</span></div>';
    return html;
}

/**
 * Display Phase 4: Encoder
 */
function displayEncoder(tokens) {
    const seqLen = tokens.length;

    // Update all shape elements
    const attnInputShape = document.getElementById('attnInputShape');
    if (attnInputShape) attnInputShape.textContent = `(${seqLen}, 512)`;

    const qShape = document.getElementById('qShape');
    if (qShape) qShape.textContent = `(${seqLen}, 8, 64)`;

    const kShape = document.getElementById('kShape');
    if (kShape) kShape.textContent = `(${seqLen}, 8, 64)`;

    const vShape = document.getElementById('vShape');
    if (vShape) vShape.textContent = `(${seqLen}, 8, 64)`;

    const scoresShape = document.getElementById('scoresShape');
    if (scoresShape) scoresShape.textContent = `(${seqLen}, ${seqLen})`;

    const concatShape = document.getElementById('concatShape');
    if (concatShape) concatShape.textContent = `(${seqLen}, 512)`;

    const attnOutputShape = document.getElementById('attnOutputShape');
    if (attnOutputShape) attnOutputShape.textContent = `(${seqLen}, 512)`;

    const encoderOutputShape = document.getElementById('encoderOutputShape');
    if (encoderOutputShape) encoderOutputShape.textContent = `(${seqLen}, 512)`;

    // Q, K, V matrices
    displayQKV('queryMatrix', tokens, 'Q');
    displayQKV('keyMatrix', tokens, 'K');
    displayQKV('valueMatrix', tokens, 'V');

    // Attention heatmap
    displayAttentionHeatmap('attentionHeatmap', tokens, tokens);

    // Encoder output
    const encoderOutput = document.getElementById('encoderOutput');
    encoderOutput.innerHTML = tokens.map(t => `
        <div class="matrix-row">
            <span class="matrix-cell highlight">${truncate(t, 6)}</span>
            ${Array(5).fill(0).map(() =>
        `<span class="matrix-cell">${(Math.random() * 2 - 1).toFixed(2)}</span>`
    ).join('')}
            <span class="matrix-cell">...</span>
        </div>
    `).join('');
}

/**
 * Display Q, K, V matrices
 */
function displayQKV(elementId, tokens, type) {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.innerHTML = tokens.slice(0, 3).map(t => `
        <div class="matrix-row" style="font-size: 0.65rem;">
            ${Array(4).fill(0).map(() =>
        (Math.random() * 2 - 1).toFixed(1)
    ).join(' ')} ...
        </div>
    `).join('') + '<div>⋮</div>';
}

/**
 * Display attention heatmap
 */
function displayAttentionHeatmap(elementId, rowTokens, colTokens) {
    const element = document.getElementById(elementId);
    if (!element) return;

    // Generate attention scores
    const scores = generateAttentionScores(rowTokens.length, colTokens.length);

    // Header row
    let html = `<div class="heatmap-row">
        <span class="heatmap-label"></span>
        ${colTokens.map(t => `<span class="heatmap-label">${truncate(t, 5)}</span>`).join('')}
    </div>`;

    // Data rows
    scores.forEach((row, i) => {
        html += `<div class="heatmap-row">
            <span class="heatmap-label">${truncate(rowTokens[i], 5)}</span>
            ${row.map(v => {
            const intensity = v;
            const bg = `rgba(99, 102, 241, ${intensity})`;
            return `<span class="heatmap-cell" style="background: ${bg}">${v.toFixed(2)}</span>`;
        }).join('')}
        </div>`;
    });

    element.innerHTML = html;
}

/**
 * Generate realistic attention scores
 */
function generateAttentionScores(rows, cols) {
    const scores = [];
    for (let i = 0; i < rows; i++) {
        const row = [];
        let sum = 0;

        for (let j = 0; j < cols; j++) {
            let score = Math.random() * 0.3;
            if (i === j) score += 0.3;
            if (Math.abs(i - j) <= 1) score += 0.1;
            row.push(score);
            sum += score;
        }

        scores.push(row.map(v => v / sum));
    }
    return scores;
}

/**
 * Display Phase 5: Decoder
 */
function displayDecoder(sourceTokens, targetTokens) {
    const srcLen = sourceTokens.length;
    const tgtLen = targetTokens.length;

    // Update shapes
    const decInputShape = document.getElementById('decInputShape');
    if (decInputShape) decInputShape.textContent = `(${tgtLen}, 512)`;

    const maskShape = document.getElementById('maskShape');
    if (maskShape) maskShape.textContent = `(${tgtLen}, ${tgtLen})`;

    const crossQShape = document.getElementById('crossQShape');
    if (crossQShape) crossQShape.textContent = `(${tgtLen}, 512)`;

    const crossKVShape = document.getElementById('crossKVShape');
    if (crossKVShape) crossKVShape.textContent = `(${srcLen}, 512)`;

    const crossScoresShape = document.getElementById('crossScoresShape');
    if (crossScoresShape) crossScoresShape.textContent = `(${tgtLen}, ${srcLen})`;

    // Causal mask
    displayCausalMask(targetTokens);

    // Cross-attention
    const decoderQuery = document.getElementById('decoderQuery');
    if (decoderQuery) decoderQuery.textContent = targetTokens.join(' ');

    const encoderKV = document.getElementById('encoderKV');
    if (encoderKV) encoderKV.textContent = sourceTokens.join(' ');

    // Cross-attention heatmap
    displayAttentionHeatmap('crossAttentionHeatmap', targetTokens, sourceTokens);
}

/**
 * Display causal mask
 */
function displayCausalMask(tokens) {
    const maskElement = document.getElementById('causalMask');
    if (!maskElement) return;

    const size = Math.min(5, tokens.length);

    let html = `<div class="mask-row">
        <span class="mask-cell" style="background: transparent"></span>
        ${Array(size).fill(0).map((_, i) =>
        `<span class="mask-cell" style="background: transparent; font-size: 0.7rem">${i}</span>`
    ).join('')}
    </div>`;

    for (let i = 0; i < size; i++) {
        html += `<div class="mask-row">
            <span class="mask-cell" style="background: transparent; font-size: 0.7rem">${i}</span>
            ${Array(size).fill(0).map((_, j) => {
            const allowed = j <= i;
            return `<span class="mask-cell ${allowed ? 'allowed' : 'blocked'}">${allowed ? '✓' : '✗'}</span>`;
        }).join('')}
        </div>`;
    }

    maskElement.innerHTML = html;
}

/**
 * Display Phase 6: Output
 */
function displayOutput(sourceTokens, targetTokens) {
    const srcLen = sourceTokens.length;
    const tgtLen = targetTokens.length;

    // Update shapes in output flow
    const decOutShape = document.getElementById('decOutShape');
    if (decOutShape) decOutShape.textContent = `(${tgtLen}, 512)`;

    const logitsShape = document.getElementById('logitsShape');
    if (logitsShape) logitsShape.textContent = `(${tgtLen}, vocab)`;

    const probsShape = document.getElementById('probsShape');
    if (probsShape) probsShape.textContent = `(${tgtLen}, vocab)`;

    // Predicted token
    const predictions = ['il', 'gatto', 'sedeva', 'sul'];
    const predictedToken = predictions[Math.floor(Math.random() * predictions.length)];
    const predictedEl = document.getElementById('predictedToken');
    if (predictedEl) predictedEl.textContent = predictedToken;

    // Probability bars
    const probs = [
        { token: predictedToken, prob: 0.45 + Math.random() * 0.3 },
        { token: 'la', prob: 0.1 + Math.random() * 0.1 },
        { token: 'lo', prob: 0.05 + Math.random() * 0.05 },
        { token: 'un', prob: 0.03 + Math.random() * 0.03 },
        { token: 'le', prob: 0.02 + Math.random() * 0.02 }
    ].sort((a, b) => b.prob - a.prob);

    const total = probs.reduce((s, p) => s + p.prob, 0);
    probs.forEach(p => p.prob = p.prob / total);

    const probBars = document.getElementById('probBars');
    if (probBars) {
        probBars.innerHTML = probs.map(p => `
            <div class="prob-item">
                <span class="prob-token">${p.token}</span>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: ${p.prob * 100}%"></div>
                </div>
                <span class="prob-value">${(p.prob * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    }
}

/**
 * Utility: Truncate string
 */
function truncate(str, len) {
    if (str.length <= len) return str;
    return str.slice(0, len);
}
