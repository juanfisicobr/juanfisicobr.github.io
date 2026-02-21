/**
 * analise.js - Dr. Juan Carlos Teran Briceno
 * Gerencia a lógica de análise de conteúdo e destaques dinâmicos.
 */

let corpus = {}; // Armazena os textos integrais do dados.json
let currentOpenID = null; // Rastreia qual depoimento está sendo lido
let currentUR = ""; // Armazena a categoria (Unidade de Registro) atual

// 1. Lista de Unidades para a Tabela (Exemplos do Eixo 1)
const unidadesDeAnalise = [
    { id: "10P.12", ur: "Responsabilidade ética no feedback", cat: "Individualizada" },
    { id: "10P.13", ur: "Feedback atempado e acompanhamento", cat: "Individualizada" },
    { id: "3A.16", ur: "Feedback coletivo e simultâneo", cat: "Coletiva" },
    { id: "12P.29", ur: "Uso de gravação para treino do aluno", cat: "Hibridização" },
    { id: "7B.7", ur: "Feedback via pasta compartilhada", cat: "Hibridização" },
    { id: "6B.10", ur: "Autonomia na supervisão à distância", cat: "Hibridização" }
];

// 2. Carregamento inicial
document.addEventListener('DOMContentLoaded', () => {
    fetch('dados.json')
        .then(response => response.json())
        .then(data => {
            corpus = data;
            renderUnitsGrid();
        })
        .catch(err => console.error("Erro ao carregar corpus:", err));
});

// 3. Renderiza a lista de cards à esquerda
function renderUnitsGrid() {
    const container = document.getElementById('units-container');
    if (!container) return;

    container.innerHTML = unidadesDeAnalise.map(item => `
        <div class="unit-card" onclick="openLector('${item.id}', '${item.ur}')">
            <span class="id-label">CÓDIGO: ${item.id}</span>
            <div class="text-sm font-semibold text-white">${item.ur}</div>
            <div class="text-xs text-gray-500 mt-1 uppercase">${item.cat}</div>
        </div>
    `).join('');
}

// 4. Função Utilitária para proteger a Busca (Escape Regex)
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// 5. Lógica de Destaque Blindada
function applyHighlights(text, urName, searchTerm) {
    let highlighted = text;

    // 1. Destaque da Unidade de Registro (UR)
    if (urName) {
        const keywords = urName.split(/[\s()\/,.\-]+/).filter(w => w.length > 3);
        if (keywords.length > 0) {
            // Removidos os parênteses extras do RegExp para não criar grupos de captura
            const urRegex = new RegExp(keywords.join('|'), 'gi');
            highlighted = highlighted.replace(urRegex, '<span class="hl-ur">$&</span>');
        }
    }

    // 2. Destaque da Busca Global
    if (searchTerm && searchTerm.length > 2) {
        const safeSearch = escapeRegExp(searchTerm);
        // RegExp sem parênteses para manter a assinatura (match, offset, originalText)
        const searchRegex = new RegExp(safeSearch, 'gi');
        
        highlighted = highlighted.replace(searchRegex, (match, offset, originalText) => {
            const preceding = originalText.substring(0, offset);
            if (preceding.lastIndexOf('<') > preceding.lastIndexOf('>')) {
                return match; // Está dentro de uma tag HTML, não destaca
            }
            return `<span class="hl-search">${match}</span>`;
        });
    }

    return highlighted;
}

// 6. O "Lector": Abre o texto no painel lateral
function openLector(id, ur) {
    currentOpenID = id;
    currentUR = ur;
    
    const pane = document.getElementById('reading-pane');
    const body = document.getElementById('pane-body');
    const label = document.getElementById('pane-id');
    const searchInput = document.getElementById('global-search');
    const searchTerm = searchInput ? searchInput.value.trim() : "";

    if (corpus[id]) {
        label.innerText = `Depoimento na Íntegra - ${id}`;
        body.innerHTML = `“${applyHighlights(corpus[id], ur, searchTerm)}”`;
        pane.classList.remove('hidden');
        
        // Garante que o painel seja visível em telas pequenas
        if (window.innerWidth < 768) {
            pane.scrollIntoView({ behavior: 'smooth' });
        }
    }
}

// 7. Atualização dinâmica ao digitar na busca
function triggerGlobalHighlight() {
    if (currentOpenID) {
        openLector(currentOpenID, currentUR);
    }
}
