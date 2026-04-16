import random

# Executive Summary Generator with 25 Templates
def generate_executive_summary_templates(student, min_words=50, max_words=60):

    # Validate input
    skills = student.get("skills", [])
    interests = student.get("areas_of_interest", [])

    if not skills or not interests:
        return None, "Skills and areas_of_interest are required"

    # Create sample phrases
    skill_phrases = ", ".join(random.sample(skills, min(3, len(skills))))
    interest_phrases = ", ".join(random.sample(interests, min(2, len(interests))))

    # Templates (25)
    templates = [
        "Passionate about {interests}, I apply my expertise in {skills} to solve practical challenges. I combine analytical thinking with hands-on learning to strengthen my technical abilities. My goal is to contribute to impactful projects that drive meaningful results.",
        "With a strong foundation in {skills}, I actively explore opportunities in {interests}. My approach blends structured problem-solving with creativity, helping me tackle complex tasks effectively. I continuously refine my capabilities to deliver high-quality outcomes.",
        "I focus on developing strong technical skills such as {skills}, which support my interest in {interests}. Through consistent practice and curiosity, I work toward building practical solutions and contributing effectively to collaborative projects.",
        "Driven by curiosity in {interests}, I use my capabilities in {skills} to approach challenges with clarity and precision. I emphasize learning through real-world application and strive to improve my problem-solving efficiency.",
        "My work is centered around applying {skills} to areas like {interests}. I value practical thinking, continuous growth, and delivering reliable results. I aim to contribute meaningfully to projects that require structured analysis and strong technical understanding.",
        "I consistently strengthen my expertise in {skills}, motivated by an interest in {interests}. My learning approach integrates experimentation, analytical reasoning, and real-world application to build impactful solutions.",
        "Combining my strengths in {skills} with my interest in {interests}, I approach challenges with a balance of logic and creativity. I aim to produce results that are reliable, scalable, and aligned with real project needs.",
        "My interest in {interests} inspires me to apply {skills} in meaningful ways. I focus on improving my technical decision-making and building solutions that demonstrate depth, clarity, and efficiency.",
        "I apply skills such as {skills} to explore domains like {interests}. My learning philosophy emphasizes consistency, problem-solving, and practical execution. I aim to contribute effectively to innovation-focused work.",
        "Guided by a passion for {interests}, I leverage my knowledge in {skills} to analyze problems and create structured solutions. I continuously enhance my abilities to stay aligned with evolving industry expectations.",
        "My technical journey is shaped by applying {skills} across areas related to {interests}. I prioritize practical insights, systematic thinking, and outcome-driven work to build strong professional capabilities.",
        "I actively integrate my skills in {skills} with my interest in {interests}, enabling me to approach challenges with a solution-oriented mindset. My focus is on building reliable, well-structured, and impactful results.",
        "Exploring {interests} has helped me refine my abilities in {skills}. I believe in continuous improvement, and I approach tasks with a mix of analytical reasoning and hands-on experimentation.",
        "My technical strengths—such as {skills}—enable me to engage deeply with areas like {interests}. I aim to apply structured thinking and consistent practice to produce meaningful work.",
        "Driven by interest in {interests}, I build on foundational skills like {skills} to solve problems thoughtfully and efficiently. I value clarity, precision, and continuous learning as key drivers of my development.",
        "I continuously develop my abilities in {skills} while exploring concepts in {interests}. My approach blends practical experimentation with structured analysis to achieve dependable and impactful results.",
        "With experience in {skills}, I pursue opportunities in {interests} that challenge me to think critically and innovate. I focus on applying my knowledge to meaningful, real-world scenarios.",
        "My interests in {interests} motivate me to strengthen my skills in {skills}. I prioritize hands-on learning, practical reasoning, and consistent improvement to deliver well-crafted outcomes.",
        "I apply my growing expertise in {skills} to explore areas related to {interests}. I take a structured, problem-focused approach to learning, aiming to deliver solutions that demonstrate clarity and precision.",
        "Motivated by curiosity in {interests}, I reinforce my knowledge of {skills} through active practice and exploration. My focus is on building scalable, thoughtful, and well-engineered outputs.",
        "I combine understanding of {skills} with an interest in {interests} to approach challenges analytically. I work toward developing solutions that are practical, efficient, and aligned with real-world needs.",
        "My interest in {interests} guides my application of {skills}, helping me solve problems with a balanced blend of creativity and logic. I prioritize consistent learning and strong execution.",
        "By applying my abilities in {skills}, I explore topics related to {interests} with curiosity and discipline. I focus on producing outcomes that reflect thoughtful planning and technical depth.",
        "I strengthen my skill set—especially in {skills}—by engaging with ideas in {interests}. I aim to create meaningful contributions through structured thinking and continuous refinement.",
        "My passion for {interests} complements my skills in {skills}, allowing me to approach challenges with clarity and purpose. I prioritize developing solutions that are impactful, accurate, and professionally crafted."
    ]

    # Choose one template
    chosen = random.choice(templates)

    # Fill placeholders
    summary = chosen.format(skills=skill_phrases, interests=interest_phrases)

    # Enforce 50–60 words
    words = summary.split()

    # Trim
    if len(words) > max_words:
        summary = " ".join(words[:max_words])

    # Pad if short
    elif len(words) < min_words:
        filler = " I remain committed to continuous learning and applying my abilities to meaningful opportunities."
        summary += filler
        summary = " ".join(summary.split()[:max_words])

    # MUST RETURN (summary, error)
    return summary, None
