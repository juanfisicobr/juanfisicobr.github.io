/**
 * global.js - Dr. Juan Carlos Teran Briceno
 * Gerencia o carregamento do header modular e animações globais.
 */

// 1. Função para carregar o header.html em todas as páginas
async function loadHeader() {
    const placeholder = document.getElementById('header-placeholder');
    if (!placeholder) return;

    try {
        const response = await fetch('header.html');
        if (!response.ok) throw new Error('Falha ao carregar o header.');
        const html = await response.text();
        placeholder.innerHTML = html;

        // Após injetar o HTML, ativamos os eventos de clique do menu
        initHeaderInteractivity();
    } catch (error) {
        console.error('Erro ao carregar o cabeçalho:', error);
    }
}

// 2. Lógica para o menu mobile e dropdowns (extraída do seu script original)
function initHeaderInteractivity() {
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    const mobileDisciplinasButton = document.getElementById('mobile-disciplinas-button');
    const mobileDisciplinasMenu = document.getElementById('mobile-disciplinas-menu');

    // Toggle do Menu Principal Mobile
    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.onclick = () => {
            mobileMenu.classList.toggle('hidden');
        };
    }

    // Toggle do Dropdown de Disciplinas no Mobile
    if (mobileDisciplinasButton && mobileDisciplinasMenu) {
        const arrow = mobileDisciplinasButton.querySelector('svg');
        mobileDisciplinasButton.onclick = (e) => {
            e.preventDefault();
            mobileDisciplinasMenu.classList.toggle('hidden');
            if (arrow) arrow.classList.toggle('rotate-180');
        };
    }
}

// 3. Sistema de animações ao rolar a página (Fade-in)
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target); // Para a animação ocorrer apenas uma vez
            }
        });
    }, observerOptions);

    // Seleciona todos os elementos com a classe .fade-in
    document.querySelectorAll('.fade-in').forEach(el => {
        observer.observe(el);
    });
}

// 4. Inicialização ao carregar o DOM
document.addEventListener('DOMContentLoaded', () => {
    loadHeader();           // Injeta o cabeçalho
    initScrollAnimations(); // Ativa os efeitos de scroll
});
