"""arXiv category taxonomy and semantic alias resolution."""

from typing import Optional

CATEGORY_TAXONOMY: dict[str, dict[str, str]] = {
    "cs": {
        "cs.AI": "Artificial Intelligence",
        "cs.AR": "Hardware Architecture",
        "cs.CC": "Computational Complexity",
        "cs.CE": "Computational Engineering, Finance, and Science",
        "cs.CG": "Computational Geometry",
        "cs.CL": "Computation and Language (NLP)",
        "cs.CR": "Cryptography and Security",
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.CY": "Computers and Society",
        "cs.DB": "Databases",
        "cs.DC": "Distributed, Parallel, and Cluster Computing",
        "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics",
        "cs.DS": "Data Structures and Algorithms",
        "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages and Automata Theory",
        "cs.GL": "General Literature",
        "cs.GR": "Graphics",
        "cs.GT": "Computer Science and Game Theory",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science",
        "cs.MA": "Multiagent Systems",
        "cs.MM": "Multimedia",
        "cs.MS": "Mathematical Software",
        "cs.NA": "Numerical Analysis",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.NI": "Networking and Internet Architecture",
        "cs.OH": "Other Computer Science",
        "cs.OS": "Operating Systems",
        "cs.PF": "Performance",
        "cs.PL": "Programming Languages",
        "cs.RO": "Robotics",
        "cs.SC": "Symbolic Computation",
        "cs.SD": "Sound",
        "cs.SE": "Software Engineering",
        "cs.SI": "Social and Information Networks",
        "cs.SY": "Systems and Control",
    },
    "stat": {
        "stat.AP": "Applications",
        "stat.CO": "Computation",
        "stat.ME": "Methodology",
        "stat.ML": "Machine Learning",
        "stat.OT": "Other Statistics",
        "stat.TH": "Statistics Theory",
    },
    "math": {
        "math.AG": "Algebraic Geometry",
        "math.AT": "Algebraic Topology",
        "math.AP": "Analysis of PDEs",
        "math.CO": "Combinatorics",
        "math.CT": "Category Theory",
        "math.CV": "Complex Variables",
        "math.DG": "Differential Geometry",
        "math.DS": "Dynamical Systems",
        "math.FA": "Functional Analysis",
        "math.GR": "Group Theory",
        "math.NA": "Numerical Analysis",
        "math.OC": "Optimization and Control",
        "math.PR": "Probability",
        "math.ST": "Statistics Theory",
    },
    "eess": {
        "eess.AS": "Audio and Speech Processing",
        "eess.IV": "Image and Video Processing",
        "eess.SP": "Signal Processing",
        "eess.SY": "Systems and Control",
    },
    "econ": {
        "econ.EM": "Econometrics",
        "econ.GN": "General Economics",
        "econ.TH": "Theoretical Economics",
    },
    "q-bio": {
        "q-bio.BM": "Biomolecules",
        "q-bio.CB": "Cell Behavior",
        "q-bio.GN": "Genomics",
        "q-bio.MN": "Molecular Networks",
        "q-bio.NC": "Neurons and Cognition",
        "q-bio.PE": "Populations and Evolution",
        "q-bio.QM": "Quantitative Methods",
    },
    "q-fin": {
        "q-fin.CP": "Computational Finance",
        "q-fin.GN": "General Finance",
        "q-fin.PM": "Portfolio Management",
        "q-fin.RM": "Risk Management",
        "q-fin.ST": "Statistical Finance",
        "q-fin.TR": "Trading and Market Microstructure",
    },
    "physics": {
        "astro-ph": "Astrophysics",
        "cond-mat": "Condensed Matter",
        "gr-qc": "General Relativity and Quantum Cosmology",
        "hep-ex": "High Energy Physics - Experiment",
        "hep-lat": "High Energy Physics - Lattice",
        "hep-ph": "High Energy Physics - Phenomenology",
        "hep-th": "High Energy Physics - Theory",
        "math-ph": "Mathematical Physics",
        "nlin": "Nonlinear Sciences",
        "nucl-ex": "Nuclear Experiment",
        "nucl-th": "Nuclear Theory",
        "quant-ph": "Quantum Physics",
    },
}

# Maps natural language terms to arXiv category codes
SEMANTIC_ALIASES: dict[str, list[str]] = {
    # AI / ML
    "artificial intelligence": ["cs.AI"],
    "machine learning": ["cs.LG", "stat.ML"],
    "deep learning": ["cs.LG", "cs.NE"],
    "reinforcement learning": ["cs.LG", "cs.AI"],
    "natural language processing": ["cs.CL"],
    "nlp": ["cs.CL"],
    "computer vision": ["cs.CV"],
    "vision": ["cs.CV"],
    "robotics": ["cs.RO"],
    "neural networks": ["cs.NE", "cs.LG"],
    "generative ai": ["cs.LG", "cs.CL", "cs.CV"],
    "llm": ["cs.CL", "cs.AI"],
    "large language models": ["cs.CL", "cs.AI"],
    "multiagent": ["cs.MA", "cs.AI"],
    "information retrieval": ["cs.IR"],
    "search": ["cs.IR"],
    "recommendation systems": ["cs.IR", "cs.LG"],
    "speech": ["cs.SD", "eess.AS"],
    "audio": ["cs.SD", "eess.AS"],
    # Systems
    "security": ["cs.CR"],
    "cryptography": ["cs.CR"],
    "distributed systems": ["cs.DC"],
    "parallel computing": ["cs.DC"],
    "networking": ["cs.NI"],
    "databases": ["cs.DB"],
    "operating systems": ["cs.OS"],
    "software engineering": ["cs.SE"],
    "programming languages": ["cs.PL"],
    "hci": ["cs.HC"],
    "human computer interaction": ["cs.HC"],
    # Math / Theory
    "algorithms": ["cs.DS"],
    "data structures": ["cs.DS"],
    "complexity": ["cs.CC"],
    "optimization": ["math.OC", "cs.LG"],
    "probability": ["math.PR", "stat.TH"],
    "statistics": ["stat.ME", "stat.TH"],
    "combinatorics": ["math.CO"],
    # Science
    "quantum computing": ["quant-ph", "cs.ET"],
    "quantum": ["quant-ph"],
    "astrophysics": ["astro-ph"],
    "biology": ["q-bio.QM", "q-bio.GN"],
    "neuroscience": ["q-bio.NC"],
    "economics": ["econ.GN", "econ.TH"],
    "finance": ["q-fin.GN", "q-fin.CP"],
    "signal processing": ["eess.SP"],
    "image processing": ["eess.IV"],
}


def get_flat_taxonomy() -> dict[str, str]:
    """Return a flat dict of category_code -> description."""
    flat = {}
    for group_cats in CATEGORY_TAXONOMY.values():
        flat.update(group_cats)
    return flat


def resolve_categories(categories: Optional[list[str]]) -> Optional[list[str]]:
    """Resolve a mix of arXiv codes and natural language terms to valid codes.

    Examples:
        ["cs.AI"]                    -> ["cs.AI"]
        ["machine learning"]         -> ["cs.LG", "stat.ML"]
        ["cs.CV", "nlp"]             -> ["cs.CV", "cs.CL"]
        ["unknown_thing"]            -> ["unknown_thing"]  (pass-through)
    """
    if not categories:
        return None

    flat = get_flat_taxonomy()
    resolved = []

    for cat in categories:
        cat_stripped = cat.strip()

        # Already a valid arXiv category code
        if cat_stripped in flat:
            resolved.append(cat_stripped)
            continue

        # Try semantic alias (case-insensitive)
        alias_key = cat_stripped.lower()
        if alias_key in SEMANTIC_ALIASES:
            resolved.extend(SEMANTIC_ALIASES[alias_key])
            continue

        # Try partial match on alias keys
        matched = False
        for alias, codes in SEMANTIC_ALIASES.items():
            if alias_key in alias or alias in alias_key:
                resolved.extend(codes)
                matched = True
                break

        if not matched:
            resolved.append(cat_stripped)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for code in resolved:
        if code not in seen:
            seen.add(code)
            unique.append(code)

    return unique if unique else None
