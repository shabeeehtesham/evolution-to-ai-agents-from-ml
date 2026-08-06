document.addEventListener('DOMContentLoaded', () => {

    /* =====================================================================
       Certifications & Credentials dynamic rendering
       ===================================================================== */
    const certificates = [
        {
            title: "AWS Certified Developer - Associate",
            category: "cloud-devops",
            platform: "Amazon Web Services (AWS)",
            icon: "☁️",
            filename: "AWSCertifiedDeveloper-Associateendorsement.pdf",
            badge: "Credential"
        },
        {
            title: "B.Tech Degree in Computer Science & Engineering",
            category: "academic",
            platform: "B.Tech Degree",
            icon: "🎓",
            filename: "Shabee_BtechCertificate.pdf",
            badge: "Degree"
        },
        {
            title: "B.Tech Academic Marksheet",
            category: "academic",
            platform: "Academic Transcript",
            icon: "📄",
            filename: "Shbeebtechmarksheet.pdf",
            badge: "Marksheet"
        },
        {
            title: "LangChain for LLM Application Development",
            category: "genai",
            platform: "DeepLearning.AI",
            icon: "🦜",
            filename: "LangChain for LLM Application Development.pdf",
            badge: "Course"
        },
        {
            title: "Building Systems with the ChatGPT API",
            category: "genai",
            platform: "DeepLearning.AI",
            icon: "🤖",
            filename: "Building Systems with the ChatGPT API.pdf",
            badge: "Course"
        },
        {
            title: "LangChain Python Package",
            category: "genai",
            platform: "DeepLearning.AI / Coursera",
            icon: "🔗",
            filename: "LangChain Python Package.pdf",
            badge: "Course"
        },
        {
            title: "Responsible Application & Guardrails for Generative AI",
            category: "genai",
            platform: "IBM / Coursera",
            icon: "🛡️",
            filename: "Responsible Application and Guardrails for Generative AI.pdf",
            badge: "Course"
        },
        {
            title: "Deep Learning Essentials",
            category: "ml",
            platform: "DeepLearning.AI",
            icon: "🧠",
            filename: "Deep Learning Essentials.pdf",
            badge: "Course"
        },
        {
            title: "Practical Deep Learning",
            category: "ml",
            platform: "Fast.ai / Coursera",
            icon: "💡",
            filename: "Practical Deep Learning.pdf",
            badge: "Course"
        },
        {
            title: "Machine Learning Essentials",
            category: "ml",
            platform: "IBM",
            icon: "⚙️",
            filename: "Machine Learning Essentials.pdf",
            badge: "Course"
        },
        {
            title: "AWS Academy Cloud Architecting",
            category: "cloud-devops",
            platform: "AWS Academy",
            icon: "☁️",
            filename: "388_3_4793003_1705485962_AWS Course Completion Certificate.pdf",
            badge: "Course"
        },
        {
            title: "AWS Cloud Foundations Course Completion",
            category: "cloud-devops",
            platform: "AWS Academy",
            icon: "☁️",
            filename: "1404_3_4793003_1705400470_AWS Course Completion Certificate.pdf",
            badge: "Course"
        },
        {
            title: "Building Language Models in AWS",
            category: "genai",
            platform: "AWS TalentNext / Academy",
            icon: "🗣️",
            filename: "BuildingLanguageModelsinAWSTalentNext.pdf",
            badge: "Course"
        },
        {
            title: "AWS Machine Learning Terminology and Process",
            category: "ml",
            platform: "AWS Academy",
            icon: "🧠",
            filename: "ml terminologyandprocess260_3_4793003_1708305157_AWS Course Completion Certificate.pdf",
            badge: "Course"
        },
        {
            title: "AI-101 Certification for Mphasis.ai Learning Path",
            category: "genai",
            platform: "Mphasis.ai",
            icon: "🤖",
            filename: "AI-101 Certification for Mphasis.ai learning path.pdf",
            badge: "Learning Path"
        },
        {
            title: "Design Patterns in Python",
            category: "software-db",
            platform: "Coursera",
            icon: "📐",
            filename: "Design Patterns in Python.pdf",
            badge: "Course"
        },
        {
            title: "Web Development with Flask",
            category: "software-db",
            platform: "Coursera",
            icon: "🌐",
            filename: "Flask - S.pdf",
            badge: "Course"
        },
        {
            title: "SQL with MySQL",
            category: "software-db",
            platform: "Coursera",
            icon: "🐬",
            filename: "SQL with MySQL.pdf",
            badge: "Course"
        },
        {
            title: "NoSQL Databases",
            category: "software-db",
            platform: "Coursera",
            icon: "🗄️",
            filename: "NoSQL.pdf",
            badge: "Course"
        },
        {
            title: "Relational Databases & SQL",
            category: "software-db",
            platform: "Coursera",
            icon: "💾",
            filename: "Relational Databases.pdf",
            badge: "Course"
        },
        {
            title: "GitHub Copilot Essentials",
            category: "genai",
            platform: "Microsoft / Coursera",
            icon: "🤖",
            filename: "GitHub Copilot Essentials.pdf",
            badge: "Course"
        },
        {
            title: "Generative AI Overview",
            category: "genai",
            platform: "DeepLearning.AI",
            icon: "✨",
            filename: "Generative AI Overview.pdf",
            badge: "Course"
        },
        {
            title: "Large Language Model Bootcamp",
            category: "genai",
            platform: "Coursera",
            icon: "🥾",
            filename: "Large Language Model Bootcamp.pdf",
            badge: "Course"
        },
        {
            title: "Large Language Model Overview",
            category: "genai",
            platform: "DeepLearning.AI",
            icon: "📖",
            filename: "Large Language Model Overview.pdf",
            badge: "Course"
        },
        {
            title: "Linear & Logistic Regression",
            category: "ml",
            platform: "Coursera",
            icon: "📈",
            filename: "Linear & Logistic Regression.pdf",
            badge: "Course"
        },
        {
            title: "Linear Regression Models",
            category: "ml",
            platform: "Coursera",
            icon: "📉",
            filename: "Linear Regression Models.pdf",
            badge: "Course"
        },
        {
            title: "Data Science Essentials - S1",
            category: "ml",
            platform: "Coursera",
            icon: "🔬",
            filename: "Data Science Essentials - S1.pdf",
            badge: "Course"
        },
        {
            title: "Data Literacy Essentials",
            category: "ml",
            platform: "Coursera",
            icon: "📊",
            filename: "Data Literacy Essentials.pdf",
            badge: "Course"
        },
        {
            title: "Artificial Intelligence Essentials",
            category: "ml",
            platform: "IBM / Coursera",
            icon: "🧠",
            filename: "Artificial Intelligence Essentials.pdf",
            badge: "Course"
        },
        {
            title: "Exploring Artificial Intelligence",
            category: "ml",
            platform: "Coursera",
            icon: "🚀",
            filename: "Exploring Artificial Intelligence.pdf",
            badge: "Course"
        },
        {
            title: "Machine Learning Essentials for Business & Tech",
            category: "ml",
            platform: "IBM / Coursera",
            icon: "💼",
            filename: "Machine Learning Essentials for Business and Technical.pdf",
            badge: "Course"
        },
        {
            title: "Types of Machine Learning Solutions",
            category: "ml",
            platform: "IBM",
            icon: "🔬",
            filename: "Types of Machine Learning Solutions.pdf",
            badge: "Course"
        },
        {
            title: "Text Mining Essentials - S1",
            category: "ml",
            platform: "Coursera",
            icon: "📝",
            filename: "Text Mining Essentials - S1.pdf",
            badge: "Course"
        },
        {
            title: "Agile Development",
            category: "software-db",
            platform: "DeepLearning.AI / Coursera",
            icon: "🔄",
            filename: "Agile Development.pdf",
            badge: "Course"
        },
        {
            title: "Software Development Essentials",
            category: "software-db",
            platform: "Coursera",
            icon: "📦",
            filename: "Software Development Essentials.pdf",
            badge: "Course"
        },
        {
            title: "Programming Fundamentals",
            category: "software-db",
            platform: "Coursera",
            icon: "💻",
            filename: "Programming Fundamentals.pdf",
            badge: "Course"
        },
        {
            title: "Git & Version Control",
            category: "cloud-devops",
            platform: "Coursera",
            icon: "🌱",
            filename: "Git.pdf",
            badge: "Course"
        },
        {
            title: "Cloud Fundamentals",
            category: "cloud-devops",
            platform: "Coursera",
            icon: "☁️",
            filename: "Cloud Fundamentals.pdf",
            badge: "Course"
        },
        {
            title: "Python Programming - Stage 1",
            category: "software-db",
            platform: "Coursera",
            icon: "🐍",
            filename: "Python - S1.pdf",
            badge: "Course"
        },
        {
            title: "Python Programming - Stage 2",
            category: "software-db",
            platform: "Coursera",
            icon: "🐍",
            filename: "Python - S2.pdf",
            badge: "Course"
        },
        {
            title: "Python Programming - Stage 3",
            category: "software-db",
            platform: "Coursera",
            icon: "🐍",
            filename: "Python - S3.pdf",
            badge: "Course"
        },
        {
            title: "Python Best Practices - S1",
            category: "software-db",
            platform: "Coursera",
            icon: "🐍",
            filename: "Python Best Practices - S1.pdf",
            badge: "Course"
        },
        {
            title: "Python Core Programming (S1)",
            category: "software-db",
            platform: "Academy",
            icon: "🐍",
            filename: "pythonS1Certificate.pdf",
            badge: "Course"
        },
        {
            title: "Python Core Programming (S2)",
            category: "software-db",
            platform: "Academy",
            icon: "🐍",
            filename: "pythonS2Certificate.pdf",
            badge: "Course"
        },
        {
            title: "Python Core Programming (S3)",
            category: "software-db",
            platform: "Academy",
            icon: "🐍",
            filename: "pythonS3Certificate.pdf",
            badge: "Course"
        }
    ];

    const certsGrid = document.getElementById('certs-grid');
    const certSearchInput = document.getElementById('cert-search-input');
    const certEmptyState = document.getElementById('certs-empty-state');
    const filterButtons = document.querySelectorAll('#cert-category-filters .filter-btn');

    let activeCategory = 'all';
    let searchQuery = '';

    function renderCertificates() {
        if (!certsGrid) return;
        certsGrid.innerHTML = '';

        const filtered = certificates.filter(cert => {
            const matchesCategory = activeCategory === 'all' || cert.category === activeCategory;
            const matchesSearch = cert.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                                  cert.platform.toLowerCase().includes(searchQuery.toLowerCase());
            return matchesCategory && matchesSearch;
        });

        if (filtered.length === 0) {
            certsGrid.style.display = 'none';
            certEmptyState.style.display = 'block';
        } else {
            certsGrid.style.display = 'grid';
            certEmptyState.style.display = 'none';

            filtered.forEach(cert => {
                const card = document.createElement('div');
                card.className = `cert-card cat-${cert.category}`;
                card.innerHTML = `
                    <div class="cert-icon">${cert.icon}</div>
                    <div class="cert-platform">${cert.platform}</div>
                    <h3>${cert.title}</h3>
                    <div class="cert-meta">
                        <span class="cert-badge">${cert.badge}</span>
                        <span>Verifiable</span>
                    </div>
                    <a href="certificates/${encodeURIComponent(cert.filename)}" target="_blank" class="cert-btn">
                        <span>View Document</span>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                    </a>
                `;
                certsGrid.appendChild(card);
            });
        }
    }

    // Setup filter button event listeners
    filterButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            filterButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeCategory = btn.getAttribute('data-category');
            renderCertificates();
        });
    });

    // Setup search input event listener
    if (certSearchInput) {
        certSearchInput.addEventListener('input', (e) => {
            searchQuery = e.target.value;
            renderCertificates();
        });
    }

    renderCertificates();
});
