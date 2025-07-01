// Custom JavaScript for torchTextClassifiers documentation

document.addEventListener('DOMContentLoaded', function() {
    // Add copy buttons to code blocks
    addCopyButtons();
    
    // Improve navigation
    improveNavigation();
    
    // Add search enhancements
    enhanceSearch();
});

function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre');
    
    codeBlocks.forEach(function(block) {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            cursor: pointer;
            opacity: 0.7;
            transition: opacity 0.2s;
        `;
        
        button.addEventListener('click', function() {
            const code = block.querySelector('code');
            const text = code ? code.textContent : block.textContent;
            
            navigator.clipboard.writeText(text).then(function() {
                button.textContent = 'Copied!';
                setTimeout(function() {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        
        button.addEventListener('mouseenter', function() {
            button.style.opacity = '1';
        });
        
        button.addEventListener('mouseleave', function() {
            button.style.opacity = '0.7';
        });
        
        block.style.position = 'relative';
        block.appendChild(button);
    });
}

function improveNavigation() {
    // Add smooth scrolling to anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    
    anchorLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Highlight current section in navigation
    highlightCurrentSection();
}

function highlightCurrentSection() {
    const sections = document.querySelectorAll('h1, h2, h3');
    const navLinks = document.querySelectorAll('.wy-menu-vertical a');
    
    function updateActiveLink() {
        let current = '';
        
        sections.forEach(function(section) {
            const rect = section.getBoundingClientRect();
            if (rect.top <= 100) {
                current = section.id;
            }
        });
        
        navLinks.forEach(function(link) {
            link.classList.remove('current');
            if (link.getAttribute('href') === '#' + current) {
                link.classList.add('current');
            }
        });
    }
    
    window.addEventListener('scroll', updateActiveLink);
    updateActiveLink();
}

function enhanceSearch() {
    // Add keyboard shortcuts for search
    document.addEventListener('keydown', function(e) {
        // Ctrl+K or Cmd+K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('input[name="q"]');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to clear search
        if (e.key === 'Escape') {
            const searchInput = document.querySelector('input[name="q"]');
            if (searchInput && searchInput === document.activeElement) {
                searchInput.value = '';
                searchInput.blur();
            }
        }
    });
}

// Add version info to footer
function addVersionInfo() {
    const footer = document.querySelector('.rst-footer-buttons');
    if (footer) {
        const versionInfo = document.createElement('div');
        versionInfo.innerHTML = `
            <div class="footer">
                Built with ❤️ by the torchTextClassifiers team | 
                <a href="https://github.com/your-org/torch-fastText">View on GitHub</a>
            </div>
        `;
        footer.parentNode.insertBefore(versionInfo, footer.nextSibling);
    }
}

// Initialize additional features
document.addEventListener('DOMContentLoaded', function() {
    addVersionInfo();
});
