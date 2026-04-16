# Full updated recommender cell — deterministic project selection integrated
import os
import pickle
import random
import numpy as np
from urllib.parse import quote_plus
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
PICKLE_PATH = "embedding_store.pkl"
SAVE_UPDATES_TO_PICKLE = True
EMB_LOAD_TIMEOUT = 300

SIM_THRESHOLD = 0.50
TOP_K = 6
GAP_THRESHOLD = 0.78
REDUNDANCY_THRESHOLD = 0.60
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------- load pickle & model ----------------
if not os.path.exists(PICKLE_PATH):
    raise FileNotFoundError(f"Pickle not found at {PICKLE_PATH}")

with open(PICKLE_PATH, "rb") as f:
    data = pickle.load(f)

if "model_path" not in data or "skill_bank" not in data:
    raise ValueError("Pickle must contain 'model_path' and 'skill_bank' keys.")

MODEL_PATH = data["model_path"]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Saved model path not found: {MODEL_PATH}")

print("Loading SentenceTransformer model from:", MODEL_PATH)
EMB = SentenceTransformer(MODEL_PATH)
EMB_DIM = EMB.get_sentence_embedding_dimension()
print("Embedding dimension:", EMB_DIM)

# ---------------- load skill bank & vectors ----------------
SKILL_BANK = list(data["skill_bank"])
SKILL_NAMES = [s[0] for s in SKILL_BANK]
SKILL_DIFFICULTY = {s[0]: s[1] for s in SKILL_BANK}
SKILL_VECS = np.asarray(data.get("skill_vectors", None))

need_recompute = False
if SKILL_VECS is None or SKILL_VECS.shape != (len(SKILL_NAMES), EMB_DIM):
    need_recompute = True

if need_recompute:
    print("Recomputing skill vectors...")
    SKILL_VECS = EMB.encode(SKILL_NAMES, convert_to_numpy=True, show_progress_bar=True)
    if SAVE_UPDATES_TO_PICKLE:
        data["skill_vectors"] = SKILL_VECS
        with open(PICKLE_PATH, "wb") as f:
            pickle.dump(data, f)
        print("Saved recomputed vectors to pickle.")

# ---------------- templates ----------------
COURSE_TEMPLATES = [
    "Master {s}: Implement complete workflows including data preparation, model building, tuning, and validation",
    "Hands-on {s}: Build, test, and improve models using real-world datasets and structured experiments",
    "{s} Advanced Lab: Apply industry-standard evaluation, error analysis, and reporting practices",
    "End-to-End {s}: Construct production-style pipelines with preprocessing, model selection, and optimization",
    "{s} Practical Foundations: Deep-dive into concepts with guided labs, tasks, and checkpoints",
    "{s} Engineer Track: Learn applied techniques used in top engineering teams",
    "{s} Operational Skills: Build repeatable experimentation systems and comparison reports",
    "Real-World {s} Applications: Solve domain problems with measurable outcomes",
    "{s} Intensive Workshop: Perform complex tasks involving data insights, diagnostics, and refinement",
    "Structured {s} Training: Build, evaluate, and iteratively improve working models",
]

PROJECT_TEMPLATES = [
    "Build an end-to-end system demonstrating how {s} solves a measurable real-world problem",
    "Develop a deployable prototype integrating {s} as the core intelligence or automation module",
    "Create a performance dashboard visualizing insights, trends, or predictions generated using {s}",
    "Design a multi-stage workflow where {s} handles classification, ranking, recommendation, or decision-making",
    "Implement a real dataset experiment showing measurable improvement achieved through {s}",
    "Build an interactive application showcasing a practical real-world use case of {s}",
    "Develop a smart assistant or chatbot powered by {s} tailored for a specific domain or user need",
    "Create a domain-specific analytics engine using {s} to derive actionable insights",
    "Construct a scalable API or microservice exposing {s}-powered functionality",
    "Design an intelligent search or retrieval system enhanced by {s}",
    "Build a knowledge graph or mapping engine that organizes concepts using {s}",
    "Create a workflow automation tool where {s} optimizes repetitive or time-consuming tasks",
    "Implement a predictive system that uses {s} to estimate future outcomes or trends",
    "Develop a recommendation engine leveraging {s} for personalization",
    "Build a simulation model where {s} governs rules, predictions, or behaviours",
    "Create a real-time monitoring dashboard enhanced with {s}-powered alerts",
    "Design a secure authentication or validation system augmented by {s}",
    "Develop a comparative scoring tool showing how {s} improves accuracy or performance",
    "Build a dynamic roadmap or learning path generator powered by {s}",
    "Implement a document processing pipeline using {s} for extraction, classification, or summarization",
    "Create a gamified quiz or assessment tool where scoring or feedback is driven by {s}",
    "Develop a modular, reusable component demonstrating how {s} integrates into larger systems",
    "Build a domain-centric visualization tool where {s} enhances understanding of key patterns",
    "Create a personalized portfolio or profile generator using {s} to craft dynamic content",
    "Develop a multi-domain project generator where {s} automatically produces structured project ideas"
]

RESEARCH_TEMPLATES = [
    "Analyze limitations, edge cases, and failure modes of modern {s} methods",
    "Perform a comparative evaluation of multiple {s} approaches using controlled experiments",
    "Investigate interpretability and explainability frameworks relevant to {s} outputs",
    "Survey next-generation techniques pushing the boundaries of {s} research",
    "Study robustness, bias, drift, and generalization challenges in advanced {s} systems",
]

# ---------------- helper functions ----------------
def slugify(skill_str: str) -> str:
    return quote_plus(skill_str.strip().lower())

def persist_skillbank_and_vectors():
    if not SAVE_UPDATES_TO_PICKLE:
        return
    try:
        data["skill_bank"] = SKILL_BANK
        data["skill_vectors"] = SKILL_VECS
        with open(PICKLE_PATH, "wb") as f:
            pickle.dump(data, f)
        print("[INFO] Persisted updated skill bank and vectors.")
    except Exception as e:
        print("[WARN] Failed:", e)

def auto_difficulty_for(skill: str) -> int:
    s = skill.lower()
    words = s.split()
    if any(tok in s for tok in ["advanced", "expert", "master", "quantum", "autonomous", "multimodal", "graph"]):
        return 5
    if len(words) >= 3:
        return 4
    if len(words) == 2:
        return 3
    if len(words) == 1 and len(s) <= 4:
        return 2
    return 3

def ensure_skill_exists(skill: str):
    global SKILL_BANK, SKILL_NAMES, SKILL_DIFFICULTY, SKILL_VECS

    skill_clean = skill.strip().lower()
    if skill_clean in SKILL_NAMES:
        return

    print(f"[INFO] Adding new skill: '{skill_clean}'")
    diff = auto_difficulty_for(skill_clean)

    SKILL_BANK.append((skill_clean, diff))
    SKILL_NAMES.append(skill_clean)
    SKILL_DIFFICULTY[skill_clean] = diff

    vec = EMB.encode([skill_clean], convert_to_numpy=True)[0]
    if SKILL_VECS is None:
        SKILL_VECS = np.array([vec])
    else:
        SKILL_VECS = np.vstack([SKILL_VECS, vec])

    persist_skillbank_and_vectors()

# ---------------- achievements logic ----------------
def extract_achievement_domains(achievements):
    domains = []
    for a in achievements or []:
        if a.get("domain"):
            domains.append(a["domain"].strip().lower())
    return domains

# ---------------- recommender helpers ----------------
def map_interest(aoi, top_k=TOP_K, threshold=SIM_THRESHOLD):
    ensure_skill_exists(aoi)
    vec = EMB.encode([aoi], convert_to_numpy=True)[0]
    sims = cosine_similarity([vec], SKILL_VECS)[0]
    idxs = np.argsort(sims)[::-1]
    out = []
    for i in idxs:
        if sims[i] < threshold:
            continue
        out.append((SKILL_NAMES[i], float(sims[i])))
        if len(out) >= top_k:
            break
    return out

def detect_gaps(student_skills, mapped_skills, threshold=GAP_THRESHOLD):
    for s in student_skills or []:
        ensure_skill_exists(s)
    if not mapped_skills:
        return []
    if not student_skills:
        return mapped_skills
    svecs = EMB.encode(student_skills, convert_to_numpy=True)
    mvecs = EMB.encode(mapped_skills, convert_to_numpy=True)
    gaps = []
    for i, skill in enumerate(mapped_skills):
        sim = cosine_similarity([mvecs[i]], svecs)[0]
        if float(np.max(sim)) < threshold:
            gaps.append(skill)
    return gaps

def suppress_similar(mapped_skills, known_skills, threshold=REDUNDANCY_THRESHOLD):
    for s in known_skills or []:
        ensure_skill_exists(s)
    if not known_skills:
        return mapped_skills
    mvecs = EMB.encode(mapped_skills, convert_to_numpy=True)
    kvecs = EMB.encode(known_skills, convert_to_numpy=True)
    final = []
    for i, s in enumerate(mapped_skills):
        drop = False
        for j, k in enumerate(known_skills):
            sim = cosine_similarity([mvecs[i]], [kvecs[j]])[0][0]
            if sim >= threshold:
                drop = True
                break
        if not drop:
            final.append(s)
    return final

# ---------------- content generation ----------------
def generate_courses(skill):
    s = skill.strip()
    slug = slugify(s)
    chosen = random.sample(COURSE_TEMPLATES, 3)
    course_list = []
    for t in chosen:
        desc = t.format(s=s.title())
        ref = f"References: https://www.coursera.org/search?query={slug}, https://www.edx.org/search?q={slug}, https://www.udemy.com/courses/search/?q={slug}"
        course_list.append(f"{desc} | {ref}")
    return course_list

def generate_projects(skill, aoi=None, top_n=3, sim_threshold=0.4):
    """
    Production-ready dynamic project generator.

    Parameters:
    -----------
    skill : str - primary skill or concept
    aoi   : str - area of interest / domain context
    top_n : int - number of project suggestions to return
    sim_threshold : float - minimum similarity to consider a template relevant

    Returns:
    --------
    List of formatted project titles (top_n)
    """
    s = skill.strip()
    domain = aoi.strip() if aoi else None

    # Step 1: Embed all templates (once in production, cache this)
    template_texts = PROJECT_TEMPLATES
    template_vecs = EMB.encode(template_texts, convert_to_numpy=True)

    # Step 2: Embed skill + AOI for semantic relevance
    query_text = f"{s} {domain}" if domain else s
    skill_vec = EMB.encode([query_text], convert_to_numpy=True)[0]

    # Step 3: Compute similarity and sort
    sims = cosine_similarity([skill_vec], template_vecs)[0]
    sorted_idxs = sims.argsort()[::-1]

    # Step 4: Select top templates above threshold
    selected_projects = []
    for idx in sorted_idxs:
        if sims[idx] < sim_threshold:
            continue
        proj_title = PROJECT_TEMPLATES[idx].format(s=s)
        if domain:
            # inject domain naturally
            proj_title = proj_title.replace("practical", f"{domain}-specific practical")
        selected_projects.append(proj_title)
        if len(selected_projects) >= top_n:
            break

    # Step 5: fallback if no high-sim templates
    if not selected_projects:
        for t in PROJECT_TEMPLATES[:top_n]:
            proj_title = t.format(s=s)
            if domain:
                proj_title = proj_title.replace("practical", f"{domain}-specific practical")
            selected_projects.append(proj_title)

    return selected_projects

def generate_research(skill):
    s = skill.strip()
    slug = slugify(s)
    chosen = random.sample(RESEARCH_TEMPLATES, 3)
    return [f"{t.format(s=s)} | References: https://scholar.google.com/scholar?q={slug}, https://arxiv.org/search/?query={slug}&searchtype=all" for t in chosen]

def explain_skill(skill, gap, student_skills):
    if not gap:
        return f"You already have exposure to '{skill}'. Focus on deeper applications and evaluation."
    base = ", ".join(student_skills) if student_skills else "your foundation"
    difficulty = SKILL_DIFFICULTY.get(skill, 1)
    return f"Start learning '{skill}' (difficulty {difficulty}) building on {base}. Complete 1 structured course + 1 measurable project."

# ---------------- FINAL recommender ----------------
def recommend(student):
    achievements = student.get("achievements", [])
    achievement_domains = extract_achievement_domains(achievements)
    all_student_skills = (student.get("skills") or []) + achievement_domains
    for s in all_student_skills + (student.get("interests") or []):
        if s:
            ensure_skill_exists(s)
    out = {"student_id": student.get("id", "anonymous"), "recommendations": []}
    for aoi in student.get("interests", []):
        mapped = map_interest(aoi)
        mapped_skills = [s for s, _ in mapped]
        scores = {s: sc for s, sc in mapped}
        gaps = detect_gaps(all_student_skills, mapped_skills)
        known = [s for s in mapped_skills if s not in gaps]
        final = suppress_similar(mapped_skills, known)
        final.sort(key=lambda x: SKILL_DIFFICULTY.get(x, 1))
        blocks = []
        for s in final:
            gap = s in gaps
            difficulty = SKILL_DIFFICULTY.get(s, 1)
            courses = generate_courses(s) if gap else []
            projects = generate_projects(s) if gap else []
            research = generate_research(s) if gap else []
            achievement_match = any(s in d for d in achievement_domains)
            blocks.append({
                "skill": s,
                "difficulty": difficulty,
                "similarity": round(scores.get(s, 0.0), 3),
                "gap": gap,
                "achievement_alignment": achievement_match,
                "course_suggestions": courses,
                "project_suggestions": projects,
                "research_suggestions": research,
                "explanation": explain_skill(s, gap, all_student_skills)
            })
        summary_courses = [b["course_suggestions"][0] for b in blocks if b["course_suggestions"]]
        summary = " → ".join(summary_courses) if summary_courses else "Focus on applied experimentation and research."
        out["recommendations"].append({
            "interest": aoi,
            "mapped_skills": final,
            "skill_blocks": blocks,
            "summary": summary
        })
    return out

# ---------------- CLI ----------------
if __name__ == "__main__":
    print("\nENTER STUDENT DETAILS")
    sid = input("Student ID: ").strip()
    dept = input("Department: ").strip()
    skills = [x.strip().lower() for x in input("Enter skills (comma-separated): ").split(",") if x.strip()]
    interests = [x.strip().lower() for x in input("Enter interests (comma-separated): ").split(",") if x.strip()]
    achievements = []
    print("\nEnter achievements as 'category,domain' — type 'done' to finish.")
    while True:
        raw = input("Achievement: ").strip()
        if raw.lower() == "done":
            break
        if "," in raw:
            cat, dom = raw.split(",", 1)
            achievements.append({"category": cat.strip(), "domain": dom.strip().lower()})
        else:
            print("Invalid format. Use: category,domain")
    STUDENT_JSON = {
        "id": sid or "anonymous",
        "department": dept,
        "skills": skills,
        "interests": interests,
        "achievements": achievements
    }
    rec = recommend(STUDENT_JSON)
    print(f"\nRecommendations for: {rec['student_id']}")
    for r in rec["recommendations"]:
        print("\nInterest:", r["interest"])
        print("Summary:", r["summary"])
        for b in r["skill_blocks"]:
            print(f"\nSkill: {b['skill']} (Difficulty: {b['difficulty']}, sim={b['similarity']})")
            print(" Gap:", b["gap"])
            print(" Achievement Alignment:", b["achievement_alignment"])
            print(" Courses:")
            for c in b["course_suggestions"]:
                print("  -", c)
            print(" Projects:")
            for p in b["project_suggestions"]:
                print("  -", p)
            print(" Research:")
            for rs in b["research_suggestions"]:
                print("  -", rs)
            print(" Explanation:", b["explanation"])
